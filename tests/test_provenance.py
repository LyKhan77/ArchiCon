import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from room_reconstruction.provenance import create_run_dir, git_snapshot, load_yaml, sha256_file, write_json


class ProvenanceTests(unittest.TestCase):
    def test_hash_matches_file_content(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.bin"
            path.write_bytes(b"room-video")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"room-video").hexdigest())

    def test_unique_run_rejects_overwrite_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = create_run_dir(root, "pilot_001")
            self.assertEqual(run, root.resolve() / "outputs/pilot_001")
            with self.assertRaises(FileExistsError):
                create_run_dir(root, "pilot_001")
            for invalid in ["../escape", "/tmp/escape", "", "a/b", "a b", ".hidden"]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    create_run_dir(root, invalid)

    def test_json_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            write_json(path, {"valid": True})
            with self.assertRaises(FileExistsError):
                write_json(path, {"valid": False})
            self.assertEqual(json.loads(path.read_text()), {"valid": True})

    def test_yaml_requires_mapping_and_rejects_python_objects(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text("schema_version: 1\npaths:\n  outputs: outputs\n")
            self.assertEqual(load_yaml(path)["schema_version"], 1)
            path.write_text("- not-a-mapping\n")
            with self.assertRaises(ValueError):
                load_yaml(path)
            path.write_text("!!python/object:builtins.object {}\n")
            with self.assertRaises(ValueError):
                load_yaml(path)

    def test_git_snapshot_records_dirty_source_but_excludes_secrets_and_media(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "src").mkdir()
            source = root / "src/example.py"
            source.write_text("VALUE = 1\n")
            (root/'docs').mkdir()
            (root/'docs/PRD.md').write_text('Canonical requirements')
            (root / "server-info.txt").write_text("password: SECRET-MUST-NOT-APPEAR\n")
            (root / "outputs").mkdir()
            (root / "outputs/large.bin").write_bytes(b"large")
            report = git_snapshot(root)
            self.assertIsNone(report["head"])
            self.assertTrue(report["dirty"])
            self.assertIn("src/example.py", report["source_files"])
            self.assertIn('docs/PRD.md',report['source_files'])
            self.assertNotIn("SECRET-MUST-NOT-APPEAR", json.dumps(report))
            self.assertNotIn("server-info.txt", report["source_files"])
            self.assertNotIn("outputs/large.bin", report["source_files"])
            old_hash = report["source_files"]["src/example.py"]["sha256"]
            source.write_text("VALUE = 2\n")
            self.assertNotEqual(old_hash, git_snapshot(root)["source_files"]["src/example.py"]["sha256"])


if __name__ == "__main__":
    unittest.main()
