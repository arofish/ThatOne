"""Pinned checkpoint identity and local validation (standard library only)."""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
# Snapshot recorded by the existing Hub download metadata. The LFS content ID
# also appears in its cached .incomplete filename; it is not inferred from size.
REVISION = "36e679e65c2a2fead755ae21162091293ad37834"
WEIGHT_SHA256 = "7b7b583c210c867410bc6bdb8a55fe14eec62999e0a9ea31ff222dc501f9cfbe"
MODEL_DIR = PROJECT_ROOT / "local-data/models/chinese-clip-base"
LOCK_FILE = MODEL_DIR.parent / "chinese-clip-base-lock.json"
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "vocab.txt", "pytorch_model.bin")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_revision(model_dir: Path, lock_file: Path) -> None:
    if lock_file.exists():
        saved = json.loads(lock_file.read_text(encoding="utf-8"))
        if saved["model_id"] != MODEL_ID or saved["revision"] != REVISION:
            raise ValueError("Existing model lock differs from the pinned checkpoint")
    state = model_dir / ".download-revision"
    if state.exists() and state.read_text(encoding="utf-8").strip() != REVISION:
        raise ValueError("Partial download belongs to a different revision")


def require_model_files(model_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES
               if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0]
    if missing:
        partial = model_dir / "pytorch_model.bin.part"
        detail = " Found a .part download; it must pass SHA-256 verification before use." if partial.exists() else ""
        raise FileNotFoundError(
            f"Model is not ready in {model_dir}; missing: {', '.join(missing)}."
            f"{detail} Run .venv-ai/bin/python scripts/download_chinese_clip_http.py first."
        )


def finalize_weight(model_dir: Path, expected_sha256: str = WEIGHT_SHA256) -> Path:
    """Promote only a verified download; preserve incomplete/corrupt files."""
    target = model_dir / "pytorch_model.bin"
    partial = model_dir / "pytorch_model.bin.part"
    if target.is_file() and sha256(target) == expected_sha256:
        return target
    if not partial.is_file():
        raise FileNotFoundError("No verified weight or .part download found")
    if sha256(partial) != expected_sha256:
        raise ValueError("Weight SHA-256 mismatch; keep .part and resume the download")
    partial.replace(target)
    return target


def write_model_lock(model_dir: Path, lock_file: Path) -> None:
    require_model_files(model_dir)
    temporary = lock_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weight_sha256": WEIGHT_SHA256,
    }, indent=2) + "\n", encoding="utf-8")
    temporary.replace(lock_file)
    (model_dir / ".download-revision").unlink(missing_ok=True)
