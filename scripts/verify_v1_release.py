
"""Inspect a local release candidate without assuming it is signed/notarized."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'local-data/v1/ThatOne-1.0.0-rc.app'
HELPER=APP/'Contents/Helpers/ThatOneBackend/ThatOneBackend'
MODEL=APP/'Contents/Resources/Models/chinese-clip-base'
manifest_path=MODEL.parent/'model-files.json'

assert HELPER.is_file(), f'Missing bundled backend executable: {HELPER}'
assert manifest_path.is_file(), f'Missing model manifest: {manifest_path}'
assert not (APP/'Contents/Resources/local-data/photoindex.db').exists(), 'Private DB copied into app'

manifest=json.loads(manifest_path.read_text())
assert manifest['model_id']=='OFA-Sys/chinese-clip-vit-base-patch16'
assert manifest['model_revision']=='36e679e65c2a2fead755ae21162091293ad37834'
for name,expected in manifest['model_files_sha256'].items():
    h=hashlib.sha256()
    with (MODEL/name).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==expected, name
print('PASS: bundled complete model hashes')
print('PASS: backend executable exists')
print('NOTE: codesign/notarization/runtime checks must be done separately on macOS')
