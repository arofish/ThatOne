"""Resume the pinned checkpoint, verify it, then make it available offline."""

import subprocess

from chinese_clip_checkpoint import (
    LOCK_FILE, MODEL_DIR, MODEL_ID, REQUIRED_FILES, REVISION, WEIGHT_SHA256,
    check_revision, finalize_weight, sha256, write_model_lock,
)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    check_revision(MODEL_DIR, LOCK_FILE)
    (MODEL_DIR / ".download-revision").write_text(REVISION, encoding="utf-8")
    print("Revision:", REVISION, flush=True)

    # A completed curl download may still have a .part suffix. Validate it
    # before contacting the network or trying curl -C on an already full file.
    try:
        finalize_weight(MODEL_DIR)
        print("Existing weight verified (no weight download needed)", flush=True)
    except (FileNotFoundError, ValueError):
        partial = MODEL_DIR / "pytorch_model.bin.part"
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/pytorch_model.bin"
        subprocess.run([
            "curl", "-fL", "--http1.1", "--retry", "3", "--retry-all-errors",
            "--retry-delay", "3", "--connect-timeout", "15",
            "--speed-time", "60", "--speed-limit", "1024",
            "-C", "-", "-o", str(partial), url,
        ], check=True)
        finalize_weight(MODEL_DIR)

    # Existing complete local files need no Hub connection. Import lazily so
    # repairing a completed download works even in a standard-library env.
    missing = [name for name in REQUIRED_FILES[:-1]
               if not (MODEL_DIR / name).is_file() or (MODEL_DIR / name).stat().st_size == 0]
    if missing:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID, revision=REVISION,
                          local_dir=str(MODEL_DIR), allow_patterns=missing)

    # Confirm the weight again before recording completion.
    if sha256(MODEL_DIR / "pytorch_model.bin") != WEIGHT_SHA256:
        raise ValueError("Weight changed during download finalization")
    write_model_lock(MODEL_DIR, LOCK_FILE)
    print("Model download completed and verified!")
    print("Saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
