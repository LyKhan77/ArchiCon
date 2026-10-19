"""Cold archive safety and recovery, using real temporary files."""
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from room_reconstruction import output_archive as archive


class OutputArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = "outputs/archive/experiments_2026-10-16"
        self.manifest = self.root / self.base / "move-manifest.json"
        self.entries = []
        for name in ("old_camera", "old_depth"):
            source = self.root / "outputs" / name
            (source / "empty").mkdir(parents=True)
            (source / "result.bin").write_bytes(b"data-" + name.encode())
            self.entries.append(dict(source=f"outputs/{name}",
                                     destination=f"{self.base}/flat/{name}",
                                     file_count=1, logical_bytes=(source / "result.bin").stat().st_size))
        self.proposal = self.root / "proposal.json"
        self.write_proposal()

    def write_proposal(self):
        self.proposal.write_text(json.dumps(dict(schema_version=1, deletion_allowed=False,
            hosts=dict(local=dict(root=str(self.root), entries=self.entries,
                                  protected=["outputs/jobs/new_user_job"])))))

    def prepare(self):
        return archive.prepare(self.root, self.proposal, "local", self.base)

    def test_apply_verify_and_restore_preserve_bytes_and_empty_directories(self):
        self.prepare()
        report = archive.apply(self.root, self.manifest)
        self.assertEqual(report["archived_entries"], 2)
        for entry in self.entries:
            self.assertFalse((self.root / entry["source"]).exists())
            self.assertTrue((self.root / entry["destination"] / "empty").is_dir())
        self.assertEqual(archive.verify(self.root, self.manifest)["archived_entries"], 2)
        self.assertEqual(archive.restore(self.root, self.manifest)["restored_entries"], 2)
        self.assertEqual(archive.verify(self.root, self.manifest)["restored_entries"], 2)
        for entry in self.entries:
            self.assertTrue((self.root / entry["source"] / "empty").is_dir())
        self.assertFalse(json.loads(self.manifest.read_text())["deletion_allowed"])

    def test_invalid_source_and_destination_refused(self):
        for source, destination in (
            ("outputs/../end_product/part1", f"{self.base}/flat/part1"),
            ("end_product/part1", f"{self.base}/flat/part1"),
            ("outputs/.run-index", f"{self.base}/flat/.run-index"),
            ("outputs/dashboard_qa", f"{self.base}/flat/dashboard_qa"),
            ("outputs/jobs/new_user_job", f"{self.base}/jobs/new_user_job"),
            ("outputs/old_camera", "outputs/outside/old_camera"),
        ):
            with self.subTest(source=source, destination=destination):
                self.entries = [dict(source=source, destination=destination)]
                self.write_proposal()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.prepare()
                self.assertFalse(self.manifest.exists())

    def test_symlink_inside_tree_refused(self):
        (self.root / "outputs/old_camera/link").symlink_to(self.root / "proposal.json")
        with self.assertRaisesRegex(ValueError, "[Ss]ymlink"):
            self.prepare()
        self.assertFalse(self.manifest.exists())

    def test_source_change_refused_before_any_move(self):
        self.prepare()
        (self.root / "outputs/old_depth/result.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            archive.apply(self.root, self.manifest)
        self.assertTrue((self.root / "outputs/old_camera").exists())

    def test_source_inventory_must_match_proposal(self):
        self.entries[-1]["file_count"] = 99
        self.write_proposal()
        with self.assertRaisesRegex(RuntimeError, "proposal"):
            self.prepare()
        self.assertFalse(self.manifest.exists())

    def test_collision_never_overwrites_and_blocks_all_moves(self):
        self.prepare()
        target = self.root / self.entries[-1]["destination"]
        target.mkdir(parents=True)
        (target / "valuable").write_text("keep")
        with self.assertRaisesRegex(RuntimeError, "[Cc]ollision"):
            archive.apply(self.root, self.manifest)
        self.assertTrue((self.root / "outputs/old_camera").exists())
        self.assertEqual((target / "valuable").read_text(), "keep")

    def test_interrupted_move_resumes_without_rewriting_existing_data(self):
        self.prepare()
        real_move = archive._rename_exclusive
        calls = []
        def interrupt(source, destination):
            calls.append(str(source))
            if len(calls) == 2:
                raise RuntimeError("simulated interruption")
            real_move(source, destination)
        with patch.object(archive, "_rename_exclusive", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                archive.apply(self.root, self.manifest)
        self.assertEqual(archive.verify(self.root, self.manifest)["archived_entries"], 1)
        self.assertEqual(archive.apply(self.root, self.manifest)["archived_entries"], 2)
        self.assertEqual(archive.restore(self.root, self.manifest)["restored_entries"], 2)

    def test_changed_archive_refuses_restore_before_any_move(self):
        self.prepare()
        archive.apply(self.root, self.manifest)
        (self.root / self.entries[-1]["destination"] / "result.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            archive.restore(self.root, self.manifest)
        self.assertFalse((self.root / "outputs/old_camera").exists())

    def test_atomic_rename_refuses_existing_empty_directory(self):
        first, second = self.root / "first", self.root / "second"
        first.mkdir()
        second.mkdir()
        with self.assertRaises(FileExistsError):
            archive._rename_exclusive(first, second)
        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())

    def test_busy_controller_blocks_moves(self):
        self.prepare()
        lock = self.root / "outputs/.locks/archicon-controller.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "[Bb]usy"):
                archive.apply(self.root, self.manifest)
        self.assertTrue((self.root / "outputs/old_camera").exists())

    def test_manifest_tampering_and_wrong_workspace_refused(self):
        self.prepare()
        data = json.loads(self.manifest.read_text())
        data["entries"][0]["destination"] = "raw_videos/old_camera"
        self.manifest.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            archive.apply(self.root, self.manifest)
        self.assertTrue((self.root / "outputs/old_camera").exists())


if __name__ == "__main__":
    unittest.main()
