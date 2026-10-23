import resource
import argparse
import hashlib
import json
import platform
import sqlite3
import sys
import time

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from photoindex.retrieval import encode_images


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "local-data/retrieval"
MANIFEST_PATH = DATA_DIR / "manifest.json"
IMAGES_PATH = DATA_DIR / "images.json"

MODEL_DIR = ROOT / "local-data/models/chinese-clip-base"
MODEL_LOCK = ROOT / "local-data/models/chinese-clip-base-lock.json"

RUNS_DIR = DATA_DIR / "runs"

EXPECTED_MANIFEST_SHA = (
    "90ba5129aa1fd4b4ce462dbfd9043088"
    "bf235738ab1662edef37523803cdf496"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "local-data/photoindex.db",
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    args = parser.parse_args()

    run_id = args.run_id

    if (
        not run_id
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError("Invalid run ID")

    run_dir = RUNS_DIR / run_id

    if run_dir.exists():
        raise FileExistsError(run_dir)

    # -----------------------------------
    # 1. 校验冻结评测清单
    # -----------------------------------

    manifest_hash = sha256_file(MANIFEST_PATH)

    if manifest_hash != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Frozen manifest SHA mismatch")

    manifest = read_json(MANIFEST_PATH)

    asset_ids = manifest["asset_ids"]
    source_id = manifest["source_id"]

    if len(asset_ids) != 50:
        raise ValueError("Expected 50 Asset IDs")

    if len(set(asset_ids)) != 50:
        raise ValueError("Duplicate Asset IDs")

    if any(type(x) is not int for x in asset_ids):
        raise ValueError("Asset IDs must be integers")

    # -----------------------------------
    # 2. 校验 SQLite 里的 Asset 身份
    # -----------------------------------

    db_path = args.db.resolve(strict=True)

    with sqlite3.connect(
        db_path.as_uri() + "?mode=ro",
        uri=True,
    ) as conn:
        rows = conn.execute(
            """
            SELECT id
            FROM assets
            WHERE source_id = ?
            """,
            (source_id,),
        ).fetchall()

    db_ids = {row[0] for row in rows}

    if not set(asset_ids) <= db_ids:
        raise RuntimeError(
            "Manifest contains Assets missing from SQLite"
        )

    # -----------------------------------
    # 3. 读取现有 images.json
    # -----------------------------------

    image_mapping = read_json(IMAGES_PATH)
    items = image_mapping["images"]

    mapped_ids = [item["asset_id"] for item in items]

    if len(items) != 50:
        raise ValueError("Expected 50 PNG mappings")

    if any(type(x) is not int for x in mapped_ids):
        raise ValueError("Mapping IDs must be integers")

    if len(set(mapped_ids)) != 50:
        raise ValueError("Duplicate mapping Asset IDs")

    if set(mapped_ids) != set(asset_ids):
        raise ValueError(
            "images.json does not match manifest"
        )

    png_by_id = {
        item["asset_id"]: item["png_path"]
        for item in items
    }

    # -----------------------------------
    # 4. 按 manifest 顺序恢复 PNG 路径
    # -----------------------------------

    paths = []
    input_records = []

    for asset_id in asset_ids:
        relative = Path(png_by_id[asset_id])

        # 当前 JSON 的路径约定为项目根相对路径
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid PNG path")

        path = (ROOT / relative).resolve(strict=True)

        if not path.is_relative_to(
            (DATA_DIR / "thumbnails").resolve()
        ):
            raise ValueError("PNG outside thumbnails directory")

        if path.suffix.lower() != ".png":
            raise ValueError(f"Not a PNG: {path}")

        # 确认文件真实可解码
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Invalid PNG: {path}")

            image.load()
            width, height = image.size

        paths.append(path)

        input_records.append({
            "asset_id": asset_id,
            "png_path": relative.as_posix(),
            "width": width,
            "height": height,
            "sha256": sha256_file(path),
        })

    if len(set(paths)) != 50:
        raise ValueError("Duplicate PNG paths")

    print("Verified 50 PNG inputs")

    # -----------------------------------
    # 5. 检查模型锁与本地文件
    # -----------------------------------

    model_lock = read_json(MODEL_LOCK)

    model_revision = model_lock["revision"]

    if len(model_revision) != 40:
        raise ValueError("Invalid model revision")

    processor_config = MODEL_DIR / "preprocessor_config.json"
    weight_file = MODEL_DIR / "pytorch_model.bin"

    processor_hash = sha256_file(processor_config)
    weight_hash = sha256_file(weight_file)

    # -----------------------------------
    # 6. CPU 加载模型，只加载一次
    # -----------------------------------

    started = time.perf_counter()

    processor = ChineseCLIPProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=False,
    )

    model = ChineseCLIPModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model = model.to("cpu", dtype=torch.float32)
    model.eval()

    load_seconds = time.perf_counter() - started

    # -----------------------------------
    # 7. 一次性编码 50 张 PNG
    # -----------------------------------

    started = time.perf_counter()

    vectors = encode_images(
        paths=paths,
        model=model,
        processor=processor,
        device="cpu",
    )

    encode_seconds = time.perf_counter() - started

    # -----------------------------------
    # 8. 检查模型输出
    # -----------------------------------

    expected_shape = (
        50,
        model.config.projection_dim,
    )

    if vectors.shape != expected_shape:
        raise RuntimeError("Unexpected vector shape")

    if vectors.dtype != np.float32:
        raise RuntimeError("Expected float32")

    if not np.isfinite(vectors).all():
        raise RuntimeError("Non-finite vectors")

    norms = np.linalg.norm(vectors, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError("Vectors not normalized")

    # 确认编码期间文件没有被修改
    for path, record in zip(paths, input_records):
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(
                f"PNG changed during encoding: {record['asset_id']}"
            )

    if sha256_file(MANIFEST_PATH) != manifest_hash:
        raise RuntimeError("Manifest changed during encoding")

    # -----------------------------------
    # 9. 保存到新的 Run，防止覆盖旧结果
    # -----------------------------------

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    peak_rss_bytes = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    model_files_sha256 = {
        name: sha256_file(MODEL_DIR / name)
        for name in (
            "config.json",
            "preprocessor_config.json",
            "vocab.txt",
            "pytorch_model.bin",
        )
    }

    metadata = {
        "run_id": run_id,
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": manifest_hash,
        "images_json_sha256": sha256_file(IMAGES_PATH),
        "model_id": model_lock["model_id"],
        "model_revision": model_revision,
        "processor_config_sha256": processor_hash,
        "model_weights_sha256": weight_hash,
        "python_version": sys.version.split()[0],
        "architecture": platform.machine(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 4,
        "model_load_seconds": load_seconds,
        "image_encode_seconds": encode_seconds,
        "vector_shape": list(vectors.shape),
        "inputs": input_records,
        "peak_rss_bytes": peak_rss_bytes,
        "model_files_sha256": model_files_sha256,
    }

    # 使用临时目录写入，全部成功后才发布 Run
    with TemporaryDirectory(
        prefix=".building-",
        dir=RUNS_DIR,
    ) as tmp:
        stage = Path(tmp)

        np.save(
            stage / "image_vectors.npy",
            vectors,
            allow_pickle=False,
        )

        metadata["image_vectors_sha256"] = sha256_file(
            stage / "image_vectors.npy"
        )

        (stage / "asset_ids.json").write_text(
            json.dumps(asset_ids, indent=2),
            encoding="utf-8",
        )

        (stage / "run_metadata.json").write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # 重新加载验证文件确实写对了
        reloaded = np.load(
            stage / "image_vectors.npy",
            allow_pickle=False,
        )

        saved_ids = read_json(stage / "asset_ids.json")

        if not np.array_equal(vectors, reloaded):
            raise RuntimeError("Saved vectors mismatch")

        if saved_ids != asset_ids:
            raise RuntimeError("Saved IDs mismatch")

        if run_dir.exists():
            raise FileExistsError(run_dir)

        stage.rename(run_dir)

    print("Run:", run_id)
    print("Asset count:", len(asset_ids))
    print("Vector shape:", vectors.shape)
    print("Load seconds:", round(load_seconds, 3))
    print("Encode seconds:", round(encode_seconds, 3))
    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()
