
#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE="$ROOT/macos/thatone.png"

ASSETS="$ROOT/macos/ThatOne/ThatOne/Assets.xcassets"

DEST="$ASSETS/AppIcon.appiconset"

if [[ ! -f "$SOURCE" ]]; then
    echo "Icon source not found: $SOURCE" >&2
    exit 1
fi

if [[ ! -d "$ASSETS" ]]; then
    echo "Xcode asset catalog not found: $ASSETS" >&2
    exit 1
fi

mkdir -p "$DEST"

# Generate the standard macOS icon sizes.
for size in 16 32 128 256 512; do

    sips -s format png \
        -z "$size" "$size" \
        "$SOURCE" \
        --out "$DEST/icon_${size}x${size}.png" \
        >/dev/null

    retina=$((size * 2))

    sips -s format png \
        -z "$retina" "$retina" \
        "$SOURCE" \
        --out "$DEST/icon_${size}x${size}@2x.png" \
        >/dev/null

done

# Generate the asset catalog manifest.
python3 - "$DEST" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])

images = []

for size in (16, 32, 128, 256, 512):
    for scale in (1, 2):

        suffix = "@2x" if scale == 2 else ""

        images.append({
            "filename": (
                f"icon_{size}x{size}{suffix}.png"
            ),
            "idiom": "mac",
            "size": f"{size}x{size}",
            "scale": f"{scale}x",
        })

manifest = {
    "images": images,
    "info": {
        "author": "xcode",
        "version": 1,
    },
}

(directory / "Contents.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "PASS: AppIcon generated at $DEST"
