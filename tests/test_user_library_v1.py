
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from photoindex.user_library import UserLibraryService

class FakeModel:
    class config:
        class text_config:
            max_position_embeddings=32

class FakeProcessor:
    def tokenizer(self, query, **kwargs):
        return {"input_ids":list(range(min(5,len(query)+2)))}

class Encoders:
    def __init__(self): self.image_calls=0
    def images(self, paths,model,processor,device):
        self.image_calls+=len(paths)
        out=np.zeros((len(paths),512),np.float32)
        for i,p in enumerate(paths):
            with Image.open(p) as im:
                pix=im.convert('RGB').getpixel((0,0))
            channel=int(np.argmax(pix))
            out[i,channel]=1.
        return out
    def texts(self, texts,model,processor,device):
        out=np.zeros((len(texts),512),np.float32)
        out[:,0]=1.0
        return out

class UserLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/'pictures';self.source.mkdir()
        self.encoder=Encoders()
        self.service=UserLibraryService(
            data_root=self.root/'appdata',model=FakeModel(),processor=FakeProcessor(),
            image_encoder=self.encoder.images,text_encoder=self.encoder.texts
        )
        self.addCleanup(self.service.shutdown)
    def make_photo(self,name,color):
        p=self.source/name;p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(32,32),color).save(p)
        return p
    def run_job(self):
        job=self.service.start_import(self.source)
        for _ in range(150):
            d=self.service.job_status(job)
            if d['status'] in ('completed','failed'):
                self.assertEqual(d['status'],'completed', d)
                return d
            time.sleep(.02)
        self.fail('Job did not finish')
    def test_import_rescan_change_and_delete(self):
        a=self.make_photo('东京 1.jpg',(255,0,0))
        b=self.make_photo('2026-10-20/second.png',(0,255,0))
        result=self.run_job()
        self.assertEqual((result['encoded'],result['reused'],result['removed']),(2,0,0))
        first=self.service.search('hello')
        self.assertEqual(len(first.results),2)
        self.assertEqual(self.encoder.image_calls,2)
        self.assertEqual(self.service.thumbnail_bytes(first.results[0].asset_id)[:8],b'\x89PNG\r\n\x1a\n')
        ids={r.relative_path:r.asset_id for r in first.results}
        result=self.run_job()
        self.assertEqual((result['encoded'],result['reused'],result['removed']),(0,2,0))
        self.assertEqual(self.encoder.image_calls,2)
        self.make_photo('东京 1.jpg',(0,0,255))
        result=self.run_job()
        self.assertEqual((result['encoded'],result['reused'],result['removed']),(1,1,0))
        b.unlink()
        result=self.run_job()
        self.assertEqual((result['encoded'],result['reused'],result['removed']),(0,1,1))
        remaining=self.service.search('hello')
        self.assertEqual(len(remaining.results),1)
        self.assertEqual(remaining.results[0].asset_id,ids['东京 1.jpg'])
        self.assertEqual(self.service.library_status()['asset_count'],1)
    def test_restart_preserves_index(self):
        self.make_photo('persist.jpg',(255,0,0))
        self.run_job()
        ids=self.service.asset_ids
        second=UserLibraryService(
            data_root=self.root/'appdata',model=FakeModel(),processor=FakeProcessor(),
            image_encoder=self.encoder.images,text_encoder=self.encoder.texts,
        )
        try:
            self.assertEqual(second.asset_ids,ids)
            self.assertEqual(second.search('hello').results[0].asset_id,ids[0])
        finally:second.shutdown()

    def test_nested_source_rejected(self):
        self.make_photo('sub/a.jpg',(255,0,0))
        self.run_job()
        with self.assertRaises(ValueError):
            self.service.start_import(self.source/'sub')

    def test_modified_same_mtime_requires_reencoding(self):
        import os
        image=self.make_photo('same.jpg',(255,0,0))
        self.run_job()
        original_stat=image.stat()
        self.make_photo('same.jpg',(0,255,0))
        os.utime(image,ns=(original_stat.st_atime_ns,original_stat.st_mtime_ns))
        job=self.run_job()
        self.assertEqual(job['encoded'],1)

    def test_offline_source_keeps_index(self):
        self.make_photo('one.jpg',(255,0,0))
        self.run_job()
        self.source.rename(self.root/'unmounted')
        with self.assertRaises(FileNotFoundError): self.service.start_import(self.source)
        self.assertEqual(len(self.service.asset_ids),1)
    def test_failure_keeps_published_snapshot(self):
        self.make_photo('one.jpg',(255,0,0))
        self.run_job()
        self.encoder.images=lambda *a: (_ for _ in ()).throw(ValueError('corrupt'))
        self.service.image_encoder=self.encoder.images
        self.make_photo('two.jpg',(0,255,0))
        job=self.service.start_import(self.source)
        for _ in range(150):
            d=self.service.job_status(job)
            if d['status'] in ('completed','failed'):break
            time.sleep(.02)
        self.assertEqual(d['status'],'failed')
        self.assertEqual(len(self.service.asset_ids),1)

if __name__ == '__main__':unittest.main()
