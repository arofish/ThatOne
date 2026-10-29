
#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-app/bin/python"
OUT="$ROOT/local-data/v1"
MODEL="$ROOT/local-data/models/chinese-clip-base"
M2_META="$ROOT/local-data/retrieval/runs/chinese-clip-base-cpu-v2/run_metadata.json"
mkdir -p "$OUT"

"$PY" - <<'PY'
import hashlib,json
from pathlib import Path
root=Path.cwd()
model=root/'local-data/models/chinese-clip-base'
meta=json.loads((root/'local-data/retrieval/runs/chinese-clip-base-cpu-v2/run_metadata.json').read_text())
assert meta['model_revision']=='36e679e65c2a2fead755ae21162091293ad37834'
assert meta['model_id']=='OFA-Sys/chinese-clip-vit-base-patch16'
for name, expected in meta['model_files_sha256'].items():
    path=model/name
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==expected, name
print('PASS: all bundled model files match M2 metadata')
PY

# --onedir keeps native binaries and Python packages as separate files.
"$PY" -m PyInstaller \
  --noconfirm --clean --onedir \
  --name ThatOneBackend \
  --paths "$ROOT" \
  --collect-all torch \
  --collect-all transformers \
  --collect-all tokenizers \
  --collect-all huggingface_hub \
  --collect-all safetensors \
  --distpath "$OUT/python-dist" \
  --workpath "$OUT/pyinstaller-work" \
  --specpath "$OUT" \
  "$ROOT/scripts/thatone_backend_entry.py"

# Use this app only as an unsigned local candidate. External distribution has
# a separate nested-signing, hardened-runtime, notarization verification gate.
xcodebuild \
  -project "$ROOT/macos/ThatOne/ThatOne.xcodeproj" \
  -scheme ThatOne -configuration Release \
  -destination 'platform=macOS' \
  -derivedDataPath "$OUT/DerivedData" \
  MARKETING_VERSION=1.0.0 CURRENT_PROJECT_VERSION=1 \
  CODE_SIGNING_ALLOWED=NO build

SOURCE_APP="$OUT/DerivedData/Build/Products/Release/ThatOne.app"
DEST_APP="$OUT/ThatOne-1.0.0-rc.app"
rm -rf "$DEST_APP"  # Only removes this generated release-candidate path.
ditto "$SOURCE_APP" "$DEST_APP"

mkdir -p "$DEST_APP/Contents/Helpers" "$DEST_APP/Contents/Resources/Models"
ditto "$OUT/python-dist/ThatOneBackend" \
  "$DEST_APP/Contents/Helpers/ThatOneBackend"
ditto "$MODEL" \
  "$DEST_APP/Contents/Resources/Models/chinese-clip-base"

"$PY" - <<'PY'
import json
from pathlib import Path
root=Path.cwd()
meta=json.loads((root/'local-data/retrieval/runs/chinese-clip-base-cpu-v2/run_metadata.json').read_text())
out=root/'local-data/v1/ThatOne-1.0.0-rc.app/Contents/Resources/Models/model-files.json'
obj={
    'model_id':meta['model_id'],
    'model_revision':meta['model_revision'],
    'model_files_sha256':meta['model_files_sha256'],
}
out.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
print('Wrote model manifest:',out)
PY

echo "Release candidate: $DEST_APP"
du -sh "$DEST_APP" "$DEST_APP/Contents/Helpers" "$DEST_APP/Contents/Resources/Models"
file "$DEST_APP/Contents/Helpers/ThatOneBackend/ThatOneBackend"
