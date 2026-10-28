import subprocess
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SyncManifestTests(unittest.TestCase):
    def test_runtime_sources_are_in_sync_manifest(self):
        listed = set((ROOT / '.sync-files').read_text().splitlines())
        tracked = subprocess.check_output(
            ['git', 'ls-files', 'src/room_reconstruction', 'scripts'],
            cwd=ROOT,
            text=True,
        ).splitlines()
        omitted = [path for path in tracked if path not in listed]
        self.assertEqual(omitted, [], f'Runtime files missing from .sync-files: {omitted}')
