#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/local-data/m3/dist/ThatOne.app"
PRODUCT="$ROOT/local-data/m3/Release/ThatOne.app"

# Build the exact bundle used by the manual acceptance instructions.
# Xcode's default DerivedData product does not update this dist copy.
xcodebuild build \
    -project "$ROOT/macos/ThatOne/ThatOne.xcodeproj" \
    -scheme ThatOne \
    -configuration Release \
    -destination 'platform=macOS' \
    -derivedDataPath "$ROOT/local-data/m3/DerivedData" \
    CONFIGURATION_BUILD_DIR="$ROOT/local-data/m3/Release" \
    CODE_SIGNING_ALLOWED=NO

mkdir -p "$(dirname "$APP")"
STAGE="$(mktemp -d "$ROOT/local-data/m3/dist/.thatone-build.XXXXXX")"
trap 'rm -rf -- "$STAGE"' EXIT

# Replace the whole bundle so files from an older Debug copy cannot survive.
ditto "$PRODUCT" "$STAGE/ThatOne.app"
# Local development signing only; distribution signing is a separate milestone.
codesign --force --sign - "$STAGE/ThatOne.app"
codesign --verify --strict "$STAGE/ThatOne.app"
if [[ -e "$APP" ]]; then
    mv "$APP" "$STAGE/previous.app"
fi
if ! mv "$STAGE/ThatOne.app" "$APP"; then
    if [[ -e "$STAGE/previous.app" ]]; then
        mv "$STAGE/previous.app" "$APP"
    fi
    exit 1
fi
printf '\nBuilt and verified: %s\nQuit any running ThatOne, then run:\n' "$APP"
printf 'open "%s"\n' "$APP"
