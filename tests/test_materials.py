import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from dec_engine.loader import combine_geometry_parts, geometry_part
from dec_engine.materials import Material
from dec_engine.project import load_project, save_project
from dec_engine.scene import MeshInstance, Scene


def make_mesh(name="Mesh"):
    return MeshInstance(
        name=name,
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32
        ),
        normals=np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=np.float32),
        uvs=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        indices=np.asarray((0, 1, 2), dtype=np.int32),
        color=(1.0, 1.0, 1.0),
    )


def part(material, offset=0.0, faces=((0, 1, 2),), texture=None):
    vertices = np.asarray(
        ((offset, 0.0, 0.0), (offset + 1.0, 0.0, 0.0), (offset, 1.0, 0.0)),
        dtype=np.float32,
    )
    return {
        "vertices": vertices,
        "normals": np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=np.float32),
        "uvs": np.zeros((3, 2), dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int32),
        "material": material,
        "texture_image": texture,
    }


class MaterialModelTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        material = Material()

        self.assertEqual(material.base_color, (1.0, 1.0, 1.0))
        self.assertEqual(material.alpha_mode, "OPAQUE")
        self.assertEqual(material.to_dict()["roughness"], 1.0)

    def test_invalid_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "name"):
            Material(name="   ")
        with self.assertRaisesRegex(ValueError, "base_color"):
            Material(base_color=(2.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "metallic"):
            Material(metallic=1.5)
        with self.assertRaisesRegex(ValueError, "alpha_mode"):
            Material(alpha_mode="TRANSLUCENT")
        with self.assertRaisesRegex(ValueError, "base_color_texture"):
            Material(base_color_texture="")

    def test_dict_round_trip_preserves_values(self):
        original = Material(
            name="Metal",
            base_color=(0.2, 0.4, 0.6),
            metallic=0.9,
            roughness=0.25,
            emissive=(0.1, 0.0, 0.0),
            alpha_mode="BLEND",
            base_color_texture="metal.png",
        )

        restored = Material.from_dict(original.to_dict())

        self.assertEqual(restored, original)

    def test_from_dict_rejects_invalid_payloads(self):
        with self.assertRaisesRegex(ValueError, "mapping"):
            Material.from_dict([])
        with self.assertRaisesRegex(ValueError, "Invalid material data"):
            Material.from_dict({"metallic": "high"})

    def test_with_texture_validates_field(self):
        material = Material().with_texture("normal_texture", "normal.png")

        self.assertEqual(material.normal_texture, "normal.png")
        with self.assertRaisesRegex(ValueError, "Unknown material texture field"):
            Material().with_texture("bump_texture", "bump.png")


class SubmeshCombinationTests(unittest.TestCase):
    def test_parts_are_merged_with_face_material_indices(self):
        first = Material(name="Wood", base_color=(0.6, 0.4, 0.2))
        second = Material(name="Metal", metallic=1.0, roughness=0.2)

        mesh = combine_geometry_parts(
            [part(first), part(second, offset=1.0)], "Combined"
        )

        self.assertEqual(len(mesh.vertices), 6)
        self.assertEqual(len(mesh.indices), 6)
        np.testing.assert_array_equal(mesh.indices, (0, 1, 2, 3, 4, 5))
        np.testing.assert_array_equal(mesh.material_indices, (0, 1))
        self.assertEqual(
            [material.name for material in mesh.materials], ["Wood", "Metal"]
        )
        self.assertEqual(mesh.color, first.base_color)

    def test_texture_report_mentions_primary_texture(self):
        texture = Image.new("RGB", (2, 2), color=(255, 0, 0))
        report: list[str] = []

        mesh = combine_geometry_parts(
            [part(Material(name="A"), texture=texture)],
            "Textured",
            report=report,
        )

        self.assertIsNotNone(mesh.texture_image)
        self.assertTrue(any("основная текстура" in line for line in report))
        self.assertEqual(mesh.material_report, tuple(report))

    def test_multiple_textures_are_reported_as_loss(self):
        first = Image.new("RGB", (2, 2), color=(255, 0, 0))
        second = Image.new("RGB", (2, 2), color=(0, 255, 0))
        report: list[str] = []

        combine_geometry_parts(
            [
                part(Material(name="A"), texture=first),
                part(Material(name="B"), offset=1.0, texture=second),
            ],
            "Multi",
            report=report,
        )

        self.assertTrue(any("встроенных текстур" in line for line in report))

    def test_empty_parts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no renderable geometry"):
            combine_geometry_parts([], "Empty")

    def test_geometry_part_extracts_default_material(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64),
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
            process=False,
        )
        report: list[str] = []

        extracted = geometry_part(mesh, 0, report)

        self.assertIsNotNone(extracted)
        if extracted is None:
            self.fail("geometry_part returned no data")
        self.assertEqual(extracted["material"].name, "Part 0")
        self.assertTrue(report)


class ProjectMaterialTests(unittest.TestCase):
    def _round_trip(self, mesh):
        directory = tempfile.mkdtemp()
        path = Path(directory) / "materials.DE"
        scene = Scene()
        scene.add_mesh(mesh)
        save_project(path, scene)
        return load_project(path)

    def test_materials_and_indices_survive_save_and_load(self):
        mesh = make_mesh()
        mesh.materials = (
            Material(name="Wood", base_color=(0.5, 0.3, 0.1)),
            Material(name="Metal", metallic=0.8),
        )
        mesh.material_indices = np.asarray((0, 1, 0), dtype=np.int32)
        mesh.indices = np.asarray((0, 1, 2, 1, 2, 0, 2, 0, 1), dtype=np.int32)

        restored = self._round_trip(mesh)

        loaded = restored.objects[0]
        self.assertEqual(
            [material.name for material in loaded.materials], ["Wood", "Metal"]
        )
        np.testing.assert_array_equal(loaded.material_indices, (0, 1, 0))

    def test_legacy_object_without_materials_loads_with_empty_tuple(self):
        restored = self._round_trip(make_mesh())

        loaded = restored.objects[0]
        self.assertEqual(loaded.materials, ())
        self.assertIsNone(loaded.material_indices)

    def _tamper(self, mutate_manifest=None, mutate_arrays=None):
        directory = tempfile.mkdtemp()
        path = Path(directory) / "tampered.DE"
        scene = Scene()
        scene.add_mesh(make_mesh())
        save_project(path, scene)
        with zipfile.ZipFile(path, "r") as archive:
            manifest = json.loads(archive.read("scene.json"))
            mesh_bytes = archive.read("meshes/0.npz")
            other = {
                name: archive.read(name)
                for name in archive.namelist()
                if name not in ("scene.json", "meshes/0.npz")
            }
        with np.load(io.BytesIO(mesh_bytes), allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        if mutate_manifest is not None:
            mutate_manifest(manifest)
        if mutate_arrays is not None:
            mutate_arrays(arrays)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("scene.json", json.dumps(manifest))
            archive.writestr("meshes/0.npz", buffer.getvalue())
            for name, payload in other.items():
                archive.writestr(name, payload)
        return path

    def test_material_index_out_of_range_is_rejected(self):
        def mutate(arrays):
            arrays["material_indices"] = np.asarray((1,), dtype=np.int32)

        def manifest_mutate(manifest):
            manifest["objects"][0]["materials"] = [Material(name="Only").to_dict()]

        path = self._tamper(manifest_mutate, mutate)

        with self.assertRaisesRegex(ValueError, "index out of range"):
            load_project(path)

    def test_material_index_shape_mismatch_is_rejected(self):
        def mutate(arrays):
            arrays["material_indices"] = np.asarray((0, 0), dtype=np.int32)

        path = self._tamper(None, mutate)

        with self.assertRaisesRegex(ValueError, "expected 1 faces"):
            load_project(path)

    def test_non_list_materials_payload_is_rejected(self):
        def mutate(manifest):
            manifest["objects"][0]["materials"] = "not-a-list"

        path = self._tamper(mutate, None)

        with self.assertRaisesRegex(ValueError, "Invalid materials"):
            load_project(path)


if __name__ == "__main__":
    unittest.main()
