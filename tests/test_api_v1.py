
import tempfile
import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
from photoindex.user_library import UserLibraryService
from photoindex.api_v1 import create_app

class Model:
    class config:
        class text_config:
            max_position_embeddings=32
class Proc:
    def tokenizer(self,text,**kw):return {'input_ids':[1,2,3]}

def enc_images(paths,*args):
    m=np.zeros((len(paths),512),dtype=np.float32);m[:,0]=1.0;return m

def enc_texts(texts,*args):
    m=np.zeros((len(texts),512),dtype=np.float32);m[:,0]=1.0;return m

class V1APITests(unittest.TestCase):
    def test_auth_import_search_update(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);source=root/'photos';source.mkdir()
            Image.new('RGB',(16,16),'red').save(source/'one.jpg')
            svc=UserLibraryService(data_root=root/'app',model=Model(),processor=Proc(),image_encoder=enc_images,text_encoder=enc_texts)
            try:
                app=create_app(service=svc,token='A'*43)
                auth={'Authorization':'Bearer '+'A'*43}
                with TestClient(app,raise_server_exceptions=False) as client:
                    self.assertEqual(client.get('/health').status_code,401)
                    self.assertEqual(client.get('/library/status').status_code,401)
                    self.assertEqual(client.get('/health',headers=auth).json()['asset_count'],0)
                    begin=client.post('/library/import',headers=auth,json={'directory':str(source)})
                    self.assertEqual(begin.status_code,202,begin.text)
                    jobid=begin.json()['job_id']
                    for _ in range(200):
                        job=client.get('/library/jobs/'+jobid,headers=auth).json()
                        if job['status'] in ('failed','completed'):break
                        time.sleep(.01)
                    self.assertEqual(job['status'],'completed',job)
                    self.assertEqual(client.get('/health',headers=auth).json()['asset_count'],1)
                    query=client.post('/search',headers=auth,json={'query':'test'})
                    self.assertEqual(query.status_code,200,query.text)
                    self.assertEqual(query.json()['results'][0]['asset_id'],1)
                    pic=client.get('/assets/1/thumbnail',headers=auth)
                    self.assertEqual(pic.status_code,200,pic.text[:100] if pic.status_code !=200 else '')
                    self.assertTrue(pic.content.startswith(b'\x89PNG'))
                    self.assertEqual(client.get('/assets/999/thumbnail',headers=auth).status_code,404)
            finally:svc.shutdown()

if __name__ == '__main__':unittest.main()
