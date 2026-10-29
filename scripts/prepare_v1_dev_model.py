
"""Generate an independent ignored dev model manifest from frozen M2 metadata."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'local-data/retrieval/runs/chinese-clip-base-cpu-v2/run_metadata.json'
MODEL=ROOT/'local-data/models/chinese-clip-base'
OUT=MODEL.parent/'model-files.json'
meta=json.loads(RUN.read_text(encoding='utf-8'))
assert meta['run_id']=='chinese-clip-base-cpu-v2'
assert meta['model_id']=='OFA-Sys/chinese-clip-vit-base-patch16'
assert meta['model_revision']=='36e679e65c2a2fead755ae21162091293ad37834'
for name,expected in meta['model_files_sha256'].items():
    h=hashlib.sha256()
    with (MODEL/name).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected:
        raise RuntimeError('Model hash mismatch: '+name)
obj={
 'model_id':meta['model_id'],
 'model_revision':meta['model_revision'],
 'model_files_sha256':meta['model_files_sha256'],
}
OUT.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print('PASS dev model manifest:', OUT)
