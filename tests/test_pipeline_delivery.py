import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from room_reconstruction.pipeline_delivery import verify_bundle,finalize_delivery,clone_delivery
from room_reconstruction.provenance import sha256_file


class DeliveryTests(unittest.TestCase):
    def test_review_clone_supports_new_job_delivery_without_touching_part1(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'end_product/capture/v001';source.mkdir(parents=True)
            (source/'index.html').write_bytes(b'exact reviewed product')
            copied=clone_delivery(root,source,'capture',delivery_root='end_product')
            self.assertEqual(copied,root/'end_product/capture/v002')
            self.assertEqual((copied/'index.html').read_bytes(),b'exact reviewed product')
            with self.assertRaises(ValueError):clone_delivery(root,source,'part1',delivery_root='end_product')

    def test_bundle_rejects_modified_artifact_and_path_escape(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);p=root/'file.txt';p.write_text('actual')
            bundle=dict(schema_version=1,input_sha256='video',components=[],artifacts=[dict(file='file.txt',sha256=sha256_file(p),size_bytes=6)])
            verify_bundle(root,bundle,input_sha256='video')
            p.write_text('changed')
            with self.assertRaises(ValueError):verify_bundle(root,bundle,input_sha256='video')
            bundle['artifacts'][0]['file']='../outside'
            with self.assertRaises(ValueError):verify_bundle(root,bundle,input_sha256='video')

    def test_unreviewed_delivery_never_claims_acceptance(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);p=root/'end_product/part2/job/v001';p.mkdir(parents=True)
            (p/'index.html').write_text('<html>offline bundle</html>')
            bundle=dict(job_id='job',coverage=dict(attempted=10,processed=8,gaps=['f1','f2'],full_room_reconstruction_complete=False))
            finalize_delivery(root,p,bundle,None,None)
            m=json.loads((p/'manifest.json').read_text())
            self.assertEqual(m['review_status'],'unavailable');self.assertEqual(m['quality_status'],'unassessed')
            with self.assertRaises(FileExistsError):finalize_delivery(root,p,bundle,None,None)


if __name__=='__main__':unittest.main()
