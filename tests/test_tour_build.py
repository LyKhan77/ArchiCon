"""Verify offline font packaging and source geometry preservation, using a real build."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node") and (ROOT / "web/node_modules/esbuild").exists(),
                     "Offline HTML build requires local Node/esbuild runtime")
class TourBuildTests(unittest.TestCase):
    def test_fonts_are_embedded_and_source_geometry_stays_exact(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            gltf = json.dumps(dict(asset=dict(version="2.0"), scenes=[dict(nodes=[])], scene=0)).encode()
            gltf += b" " * (-len(gltf) % 4)
            mesh = struct.pack("<4sIIII", b"glTF", 2, 20 + len(gltf), len(gltf), 0x4E4F534A) + gltf
            (folder / "mesh.glb").write_bytes(mesh)
            component = dict(id="main", name="Area utama", mesh_file="mesh.glb", bounds=[[0, 0, 0], [2, 3, 4]],
                floorplan=dict(bounds=[[0, 0], [2, 4]], segments=[], path=[]),
                cameras=[dict(frame_id="f0", time=0, position=[0, 1, 0],
                              matrix=[[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])])
            data = dict(schema_version=1, title="Test", components=[component])
            source, output = folder / "tour-data.json", folder / "index.html"
            source.write_text(json.dumps(data))
            command = ["node", str(ROOT / "web/build.mjs"), "--data", str(source), "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            html = output.read_text()
            for name in ("IBMPlexSans-Regular.woff2", "IBMPlexSans-Light.woff2"):
                encoded = base64.b64encode((ROOT / "web/dashboard/fonts" / name).read_bytes()).decode()
                self.assertTrue("data:font/woff2;base64," + encoded in html, name + " must be embedded for offline use")
            self.assertIn("SIL OPEN FONT LICENSE", html)
            self.assertNotIn("/*__FONT_", html)
            embedded = json.loads(html.split('<script id="tour-data" type="application/json">')[1].split('</script>')[0])
            actual = embedded["components"][0]
            self.assertEqual(actual["cameras"], component["cameras"])
            self.assertEqual(actual["bounds"], component["bounds"])
            self.assertEqual(actual["floorplan"], component["floorplan"])
            self.assertEqual(actual["mesh_sha256"], hashlib.sha256(mesh).hexdigest())
            self.assertEqual(json.loads(source.read_text()), data)
            original = output.read_bytes()
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
