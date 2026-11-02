import json
import unittest
import zipfile
from pathlib import Path

from dec_engine.project import (
    CURRENT_PROJECT_FORMAT_VERSION,
    load_project,
    migrate_project_manifest,
    save_project,
)
from dec_engine.scene import Scene


GOLDEN_PROJECTS = Path(__file__).parent / "fixtures" / "projects"


class ProjectMigrationTests(unittest.TestCase):
    def test_migrations_accept_every_historical_version_and_preserve_payload(self):
        for old_version in range(1, CURRENT_PROJECT_FORMAT_VERSION + 1):
            with self.subTest(version=old_version):
                old_manifest = {
                    "version": old_version,
                    "format": "DEcProject" if old_version >= 4 else "legacy",
                    "objects": [{"name": "unchanged"}],
                }
                migrated = migrate_project_manifest(old_manifest)

                self.assertEqual(
                    migrated["format_version"], CURRENT_PROJECT_FORMAT_VERSION
                )
                self.assertEqual(migrated["version"], CURRENT_PROJECT_FORMAT_VERSION)
                self.assertEqual(migrated["source_version"], old_version)
                self.assertEqual(migrated["objects"], old_manifest["objects"])
                self.assertNotIn("format_version", old_manifest)

    def test_migration_is_idempotent_for_current_and_migrated_manifests(self):
        legacy = {"version": 3, "lighting": {"intensity": 0.5}}
        migrated_once = migrate_project_manifest(legacy)
        migrated_twice = migrate_project_manifest(migrated_once)

        self.assertEqual(migrated_twice, migrated_once)
        self.assertEqual(migrated_twice["source_version"], 3)

    def test_legacy_global_lighting_migrates_into_directional_light(self):
        legacy_lighting = {
            "direction": [-0.2, -0.8, -0.4],
            "color": [0.2, 0.3, 0.4],
            "intensity": 0.35,
        }
        version_two = migrate_project_manifest(
            {"version": 2, "lighting": legacy_lighting}
        )
        self.assertEqual(len(version_two["lights"]), 1)
        self.assertEqual(version_two["lights"][0]["direction"], [-0.2, -0.8, -0.4])
        self.assertEqual(version_two["lights"][0]["color"], [0.2, 0.3, 0.4])
        self.assertEqual(version_two["lights"][0]["intensity"], 0.35)

        version_three = migrate_project_manifest(
            {
                "version": 3,
                "lighting": legacy_lighting,
                "lights": [
                    {
                        "node_id": "point-id",
                        "name": "Point",
                        "kind": "point",
                    },
                    {
                        "node_id": "sun-id",
                        "name": "Sun",
                        "kind": "directional",
                        "direction": [0.0, -1.0, 0.0],
                        "color": [1.0, 1.0, 1.0],
                        "intensity": 1.0,
                    },
                ],
            }
        )
        directional = version_three["lights"][1]
        self.assertEqual(directional["node_id"], "sun-id")
        self.assertEqual(directional["direction"], legacy_lighting["direction"])
        self.assertEqual(directional["color"], legacy_lighting["color"])
        self.assertEqual(directional["intensity"], legacy_lighting["intensity"])

    def test_version_four_does_not_apply_legacy_global_light_overrides(self):
        current_light = {
            "name": "Sun",
            "kind": "directional",
            "direction": [0.0, -1.0, 0.0],
            "color": [1.0, 1.0, 1.0],
            "intensity": 2.0,
        }
        migrated = migrate_project_manifest(
            {
                "version": 4,
                "format": "DEcProject",
                "lighting": {
                    "direction": [1.0, 0.0, 0.0],
                    "color": [0.0, 0.0, 0.0],
                    "intensity": 0.0,
                },
                "lights": [current_light],
            }
        )
        self.assertEqual(migrated["lights"], [current_light])

    def test_invalid_or_conflicting_versions_are_rejected(self):
        for manifest in (
            {"version": 0},
            {"version": True},
            {"version": 7},
            {"version": 5, "format_version": 6},
            {"version": 4, "format": "not-a-map"},
        ):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                migrate_project_manifest(manifest)

    def test_golden_legacy_maps_load_and_retain_the_mesh(self):
        for version in range(1, CURRENT_PROJECT_FORMAT_VERSION):
            path = GOLDEN_PROJECTS / f"legacy_v{version}.DE"
            with self.subTest(version=version):
                scene = load_project(path)
                self.assertEqual([obj.name for obj in scene.objects], ["Golden mesh"])
                self.assertTrue(scene.lights)

    def test_saved_maps_include_canonical_and_compatibility_version_keys(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.DE"
            save_project(path, Scene())
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("scene.json"))

        self.assertEqual(manifest["format_version"], CURRENT_PROJECT_FORMAT_VERSION)
        self.assertEqual(manifest["version"], CURRENT_PROJECT_FORMAT_VERSION)


if __name__ == "__main__":
    unittest.main()
