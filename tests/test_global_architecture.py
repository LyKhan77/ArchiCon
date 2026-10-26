import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from room_reconstruction import global_architecture
from room_reconstruction.provenance import sha256_file
from tests.test_alignment import cameras, transform


@unittest.skipUnless(importlib.util.find_spec('trimesh'), 'Trimesh geometry runtime required')
class GlobalArchitectureTests(unittest.TestCase):
    def fixture(self, root, shared=True, isolated_main=False):
        import trimesh
        points = np.random.default_rng(21).normal(size=(12, 3))
        similarity = transform()
        rows = []
        entries = [('main', np.eye(4), transform(1, -.6, (10, 0, -1))),
                   ('other', similarity, transform(1, .2, (-5, 2, 0)))]
        if isolated_main:
            entries = [('a', entries[0][1], entries[0][2]), ('b', entries[1][1], entries[1][2]),
                       ('main', entries[0][1], entries[0][2])]
        for identifier, coords, display in entries:
            base = root / 'outputs' / identifier; (base / 'undistorted').mkdir(parents=True)
            local = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]],
                                    vertex_colors=[[200, 50, 25, 255]] * 3, process=False)
            local.apply_transform(coords); local.apply_transform(display)
            mesh = base / 'room.glb'; local.export(mesh)
            camera = base / 'undistorted/cameras.json'
            camera.write_text(json.dumps(dict(pose_convention='OpenCV c2w; +x right, +y down, +z forward',
                frames=cameras(points, coords, prefix=('isolated' if isolated_main and identifier == 'main'
                                                       else 'f' if shared or identifier == 'main' else 'x')))))
            manifest = base / 'manifest.json'
            manifest.write_text(json.dumps(dict(stage='full_render', world_to_viewer=display.tolist(),
                input_and_source_hashes={str(camera): sha256_file(camera)},
                artifacts=[dict(file=mesh.name, sha256=sha256_file(mesh))])))
            rows.append(dict(id=identifier, name=identifier, mesh=str(mesh.relative_to(root)),
                             cameras=str(camera.relative_to(root)), render_manifest=str(manifest.relative_to(root))))
        artifacts = [dict(file=row[key], sha256=sha256_file(root / row[key]), size_bytes=(root / row[key]).stat().st_size)
                     for row in rows for key in ('mesh', 'cameras', 'render_manifest')]
        return dict(schema_version=1, components=rows, artifacts=artifacts, input_sha256='a' * 64)

    def test_merges_worlds_in_root_viewer_coordinates_preserving_colors(self):
        import trimesh
        from room_reconstruction.floorplan import colored_scene_mesh
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve(); config = self.fixture(root)
            run = root / 'outputs/global'; run.mkdir()
            report = global_architecture.build_architecture(root, run, config)
            self.assertEqual(report['status'], 'aligned_partial')
            mesh = colored_scene_mesh(trimesh.load(run / 'architecture.glb', force='scene', process=False))
            self.assertEqual(len(mesh.faces), 2)
            display = transform(1, -.6, (10, 0, -1))
            expected = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) @ display[:3, :3].T + display[:3, 3]
            for vertex in np.asarray(mesh.vertices):
                self.assertLess(np.linalg.norm(expected - vertex, axis=1).min(), 2e-6)
            self.assertTrue((np.asarray(mesh.visual.vertex_colors)[:, :3] == [200, 50, 25]).all())
            data = json.loads((run / 'undistorted/cameras.json').read_text())
            self.assertEqual(len(data['frames']), 12)
            self.assertEqual(report['component']['id'], 'architecture')
            with self.assertRaises(FileExistsError):
                global_architecture.build_architecture(root, run, config)

    def test_no_alignment_writes_evidence_without_a_dummy_mesh(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve(); config = self.fixture(root, shared=False)
            run = root / 'outputs/global'; run.mkdir()
            report = global_architecture.build_architecture(root, run, config)
            self.assertEqual(report['status'], 'no_alignment')
            self.assertIsNone(report['component'])
            self.assertFalse((run / 'architecture.glb').exists())
            self.assertEqual(json.loads((run / 'alignment.json').read_text())['fallback_components'], ['other'])

    def test_isolated_main_does_not_hide_a_validated_secondary_group(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve(); config = self.fixture(root, isolated_main=True)
            run = root/'outputs/global'; run.mkdir()
            report = global_architecture.build_architecture(root, run, config)
            self.assertEqual(report['status'], 'aligned_partial')
            self.assertEqual(report['aligned_components'], ['a', 'b'])
            self.assertEqual(report['fallback_components'], ['main'])
            evidence = json.loads((run/'alignment.json').read_text())
            self.assertEqual(evidence['requested_root_component'], 'main')
            self.assertEqual(evidence['root_component'], 'a')
            self.assertEqual(evidence['root_selection'], 'largest_validated_connected_group')

    def test_rejects_changed_inputs_invalid_viewer_transform_and_escape(self):
        for fault in ('checksum', 'transform', 'escape', 'symlink'):
            with self.subTest(fault=fault), TemporaryDirectory() as temp:
                root = Path(temp).resolve(); config = self.fixture(root)
                row = config['components'][1]
                if fault == 'checksum':
                    (root / row['mesh']).write_bytes(b'tampered')
                elif fault == 'transform':
                    manifest = root / row['render_manifest']; data = json.loads(manifest.read_text())
                    data['world_to_viewer'][0][0] = 10; manifest.write_text(json.dumps(data))
                    next(x for x in config['artifacts'] if x['file'] == row['render_manifest']).update(
                        sha256=sha256_file(manifest), size_bytes=manifest.stat().st_size)
                elif fault == 'escape':
                    row['mesh'] = '../elsewhere.glb'
                else:
                    original = root / row['mesh']; link = original.with_name('link.glb'); link.symlink_to(original)
                    row['mesh'] = str(link.relative_to(root))
                run = root / 'outputs/global'; run.mkdir()
                with self.assertRaises(ValueError):
                    global_architecture.build_architecture(root, run, config)


if __name__ == '__main__':
    unittest.main()
