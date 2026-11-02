import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from dec_engine.asset_pack import extract_asset, list_asset_pack
from dec_engine.project import load_project, save_project
from dec_engine.scene import MeshInstance, Scene

GOLDEN_PACKS = Path(__file__).parent / "fixtures" / "packs"


def make_mesh(name, *, texture_path=None, audio_path=None):
    return MeshInstance(
        name=name,
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        ),
        normals=np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=np.float32),
        uvs=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        indices=np.asarray((0, 1, 2), dtype=np.int32),
        color=(1.0, 1.0, 1.0),
        texture_path=texture_path,
        audio_path=audio_path,
    )


def rewrite_zip_without(path, *removed_names):
    with zipfile.ZipFile(path, "r") as source:
        kept = {
            name: source.read(name)
            for name in source.namelist()
            if name not in removed_names
        }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in kept.items():
            target.writestr(name, data)


class GoldenAssetPackTests(unittest.TestCase):
    def test_golden_asset_pack_lists_expected_assets(self):
        assets = list_asset_pack(GOLDEN_PACKS / "golden_v1.DE")

        self.assertEqual(
            [(item["name"], item["category"]) for item in assets],
            [("golden.obj", "Models"), ("golden.png", "Textures")],
        )

    def test_golden_asset_pack_extracts_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "golden.obj"
            extract_asset(
                GOLDEN_PACKS / "golden_v1.DE",
                "assets/models/golden.obj",
                destination,
            )

            self.assertIn(b"v 0 0 0", destination.read_bytes())


class CorruptArchiveTests(unittest.TestCase):
    def test_load_project_rejects_non_zip_archive_with_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.DE"
            path.write_bytes(b"this is not a zip archive")

            with self.assertRaisesRegex(ValueError, "not a valid .DE archive"):
                load_project(path)

    def test_list_asset_pack_rejects_non_zip_archive_with_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.DE"
            path.write_bytes(b"this is not a zip archive")

            with self.assertRaisesRegex(ValueError, "not a valid DEc asset pack"):
                list_asset_pack(path)

    def test_extract_asset_rejects_non_zip_archive_with_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.DE"
            path.write_bytes(b"this is not a zip archive")
            destination = Path(directory) / "out.obj"

            with self.assertRaisesRegex(ValueError, "not a valid DEc asset pack"):
                extract_asset(path, "assets/models/golden.obj", destination)

    def test_map_with_invalid_scene_json_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken-json.DE"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("scene.json", "{not valid json")

            with self.assertRaisesRegex(ValueError, "invalid scene.json"):
                load_project(path)


class MissingMapDependencyTests(unittest.TestCase):
    def _save_map(self, directory, mesh):
        if mesh.texture_path:
            texture = Path(directory) / "albedo.png"
            texture.write_bytes(b"\x89PNG\r\n\x1a\n")
            mesh.texture_path = str(texture)
        if mesh.audio_path:
            audio = Path(directory) / "voice.wav"
            audio.write_bytes(b"RIFF0000WAVE")
            mesh.audio_path = str(audio)
        scene = Scene()
        scene.add_mesh(mesh)
        path = Path(directory) / "map.DE"
        save_project(path, scene)
        return path

    def test_map_missing_mesh_data_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._save_map(directory, make_mesh("Golden mesh"))
            rewrite_zip_without(path, "meshes/0.npz")

            with self.assertRaisesRegex(ValueError, "missing mesh data"):
                load_project(path)

    def test_map_missing_texture_dependency_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._save_map(
                directory, make_mesh("Golden mesh", texture_path="placeholder.png")
            )
            rewrite_zip_without(path, "assets/textures/0.png")

            with self.assertRaisesRegex(ValueError, "missing texture asset"):
                load_project(path)

    def test_map_missing_audio_dependency_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._save_map(
                directory, make_mesh("Golden mesh", audio_path="placeholder.wav")
            )
            rewrite_zip_without(path, "assets/sounds/0.wav")

            with self.assertRaisesRegex(ValueError, "missing audio asset"):
                load_project(path)


class MissingPackDependencyTests(unittest.TestCase):
    def test_pack_missing_listed_asset_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken-pack.DE"
            manifest = {
                "format": "DEcAssetPack",
                "version": 1,
                "assets": [{"path": "assets/models/golden.obj"}],
            }
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("pack.json", json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "Asset is missing from pack"):
                list_asset_pack(path)


if __name__ == "__main__":
    unittest.main()
