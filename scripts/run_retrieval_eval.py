import argparse
import json
import math
import platform
import resource
import statistics
import sys
import time

from pathlib import Path
from tempfile import TemporaryDirectory

from photoindex.evaluation import evaluate_at_k
from scripts.chinese_clip_checkpoint import (
    MODEL_ID,
    REVISION,
    WEIGHT_SHA256,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "local-data/retrieval"
MODEL = ROOT / "local-data/models/chinese-clip-base"
MODEL_LOCK = ROOT / "local-data/models/chinese-clip-base-lock.json"

MANIFEST_SHA = (
    "90ba5129aa1fd4b4ce462dbfd9043088"
    "bf235738ab1662edef37523803cdf496"
)

MODEL_FILES = (
    "config.json",
    "preprocessor_config.json",
    "vocab.txt",
    "pytorch_model.bin",
)


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def p95_nearest_rank(values):
    if not values:
        raise ValueError("No latency samples")

    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1

    return ordered[index]

def verify_inputs(run):
    import numpy as np

    # 1. 冻结清单必须保持原样
    if sha256(DATA / "manifest.json") != MANIFEST_SHA:
        raise RuntimeError("Frozen manifest changed")

    manifest = read_json(DATA / "manifest.json")
    asset_ids = read_json(run / "asset_ids.json")
    metadata = read_json(run / "run_metadata.json")

    if (
        len(asset_ids) != 50
        or any(type(x) is not int for x in asset_ids)
        or len(set(asset_ids)) != 50
        or asset_ids != manifest["asset_ids"]
    ):
        raise ValueError("Invalid Asset IDs or order")

    if metadata["run_id"] != run.name:
        raise ValueError("Run ID mismatch")

    if metadata["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Dataset ID mismatch")

    if metadata["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("Cache manifest mismatch")

    if (
        metadata["device"] != "cpu"
        or metadata["dtype"] != "float32"
    ):
        raise ValueError("Expected CPU float32 cache")

    # 2. 查询数量与标签范围
    queries = manifest["queries"]

    if len(queries) != 12:
        raise ValueError("Expected 12 queries")

    if len({q["id"] for q in queries}) != 12:
        raise ValueError("Duplicate query IDs")

    answered_count = 0

    for query in queries:
        if not query["text"].strip():
            raise ValueError("Empty query")

        relevant = query["relevant_asset_ids"]

        if (
            any(type(x) is not int for x in relevant)
            or len(relevant) != len(set(relevant))
            or not set(relevant) <= set(asset_ids)
        ):
            raise ValueError("Invalid relevance labels")

        answered_count += bool(relevant)

    if answered_count != 10:
        raise ValueError("Expected 10 answered queries")

    # 3. 图片映射与实际 PNG 必须匹配缓存
    mapping_path = DATA / "images.json"

    if (
        sha256(mapping_path)
        != metadata["images_json_sha256"]
    ):
        raise RuntimeError("images.json changed")

    mapping = read_json(mapping_path)["images"]

    mapped_ids = [
        item["asset_id"]
        for item in mapping
    ]

    if (
        len(mapped_ids) != 50
        or len(set(mapped_ids)) != 50
        or set(mapped_ids) != set(asset_ids)
    ):
        raise ValueError("Invalid image mapping")

    png_by_id = {
        item["asset_id"]: item["png_path"]
        for item in mapping
    }

    records = metadata["inputs"]

    if [
        item["asset_id"] for item in records
    ] != asset_ids:
        raise ValueError("Cache input order mismatch")

    thumbnail_root = (
        DATA / "thumbnails"
    ).resolve(strict=True)

    for record in records:
        asset_id = record["asset_id"]

        if record["png_path"] != png_by_id[asset_id]:
            raise RuntimeError("PNG mapping mismatch")

        relative = Path(record["png_path"])

        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid PNG path")

        path = (ROOT / relative).resolve(strict=True)

        if not path.is_relative_to(thumbnail_root):
            raise ValueError("PNG outside thumbnails")

        if sha256(path) != record["sha256"]:
            raise RuntimeError(
                f"PNG changed: Asset {asset_id}"
            )

    # 4. 固定模型身份与文件内容
    lock = read_json(MODEL_LOCK)

    if (
        lock["model_id"] != MODEL_ID
        or lock["revision"] != REVISION
        or lock["weight_sha256"] != WEIGHT_SHA256
        or metadata["model_id"] != MODEL_ID
        or metadata["model_revision"] != REVISION
    ):
        raise RuntimeError("Model identity mismatch")

    saved_hashes = metadata.get(
        "model_files_sha256"
    )

    if saved_hashes is None:
        raise RuntimeError(
            "Cache lacks model file hashes; rebuild a new Run"
        )

    if set(saved_hashes) != set(MODEL_FILES):
        raise RuntimeError("Incomplete model hashes")

    for name in MODEL_FILES:
        if sha256(MODEL / name) != saved_hashes[name]:
            raise RuntimeError(
                f"Model file changed: {name}"
            )

    if (
        saved_hashes["pytorch_model.bin"]
        != WEIGHT_SHA256
        or metadata["model_weights_sha256"]
        != WEIGHT_SHA256
        or metadata["processor_config_sha256"]
        != saved_hashes["preprocessor_config.json"]
    ):
        raise RuntimeError("Model hash metadata mismatch")

    # 5. 检查向量内容
    vector_path = run / "image_vectors.npy"

    if (
        sha256(vector_path)
        != metadata.get("image_vectors_sha256")
    ):
        raise RuntimeError("Vector hash mismatch")

    vectors = np.load(
        vector_path,
        allow_pickle=False,
    )

    if (
        vectors.shape != (50, 512)
        or vectors.dtype != np.float32
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Invalid cached vectors")

    if not np.allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        atol=1e-4,
    ):
        raise ValueError("Vectors are not normalized")

    if list(vectors.shape) != metadata["vector_shape"]:
        raise ValueError("Vector shape metadata mismatch")

    peak = metadata.get("peak_rss_bytes")

    if type(peak) is not int or peak <= 0:
        raise ValueError("Missing image encoding peak RSS")

    if (
        not math.isfinite(metadata["image_encode_seconds"])
        or metadata["image_encode_seconds"] < 0
    ):
        raise ValueError("Invalid image encoding time")

    return manifest, metadata, asset_ids, vectors

def main():
    parser = argparse.ArgumentParser()

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

    if sys.platform != "darwin":
        raise RuntimeError(
            "This baseline uses macOS RSS units"
        )

    run = DATA / "runs" / run_id
    output = run / "evaluation-v1"

    if output.exists():
        raise FileExistsError(output)

    # 输入校验先于模型加载
    manifest, metadata, asset_ids, image_vectors = (
        verify_inputs(run)
    )

    import numpy as np
    import torch
    import transformers

    from transformers import (
        ChineseCLIPModel,
        ChineseCLIPProcessor,
    )

    from photoindex.retrieval import (
        encode_texts,
        rank_images,
    )

    # 图片和文本编码使用相同的软件版本
    current_versions = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
    }

    for key, value in current_versions.items():
        if metadata[key] != value:
            raise RuntimeError(
                f"Environment mismatch: {key}"
            )

    print("Input verification passed", flush=True)

    started = time.perf_counter()

    processor = ChineseCLIPProcessor.from_pretrained(
        MODEL,
        local_files_only=True,
        use_fast=False,
    )

    model = ChineseCLIPModel.from_pretrained(
        MODEL,
        local_files_only=True,
    )

    model = model.to(
        "cpu",
        dtype=torch.float32,
    )
    model.eval()

    load_seconds = time.perf_counter() - started

    def search(text):
        text_vectors = encode_texts(
            texts=[text],
            model=model,
            processor=processor,
            device="cpu",
        )

        if text_vectors.shape != (1, 512):
            raise RuntimeError(
                "Unexpected text vector shape"
            )

        return rank_images(
            asset_ids=asset_ids,
            image_vectors=image_vectors,
            query_vector=text_vectors[0],
        )

    queries = manifest["queries"]

    # 预热不计入热查询延迟
    for _ in range(3):
        search(queries[0]["text"])

    predictions = []
    latency_samples = []

    for query in queries:
        first_ranking = None

        for repeat in range(5):
            started = time.perf_counter()

            ranking = search(query["text"])

            elapsed = time.perf_counter() - started

            latency_samples.append({
                "query_id": query["id"],
                "repeat": repeat + 1,
                "seconds": elapsed,
            })

            ranked_ids = [
                asset_id
                for asset_id, _ in ranking
            ]

            if (
                len(ranking) != 50
                or len(set(ranked_ids)) != 50
                or set(ranked_ids) != set(asset_ids)
            ):
                raise RuntimeError(
                    "Incomplete ranking"
                )

            if ranking != sorted(
                ranking,
                key=lambda item: (-item[1], item[0]),
            ):
                raise RuntimeError(
                    "Ranking order mismatch"
                )

            if first_ranking is None:
                first_ranking = ranking
            elif ranked_ids != [
                asset_id
                for asset_id, _ in first_ranking
            ]:
                raise RuntimeError(
                    "Ranking changed across repeats"
                )

        predictions.append({
            "query_id": query["id"],
            "text": query["text"],
            "ranking": [
                {
                    "asset_id": asset_id,
                    "score": score,
                }
                for asset_id, score in first_ranking
            ],
        })

        print(
            query["id"],
            "Top-5:",
            [
                asset_id
                for asset_id, _ in first_ranking[:5]
            ],
            flush=True,
        )

    # 完整预测产生后，才读取标签计算指标
    by_id = {
        query["id"]: query
        for query in queries
    }

    answered = []
    no_match = []
    per_query = []

    for prediction in predictions:
        query = by_id[prediction["query_id"]]

        ranked_ids = [
            item["asset_id"]
            for item in prediction["ranking"]
        ]

        relevant = set(
            query["relevant_asset_ids"]
        )

        result = evaluate_at_k(
            ranked_ids,
            relevant,
            k=5,
        )

        per_query.append({
            "query_id": query["id"],
            "relevant_count": len(relevant),
            **result,
        })

        if relevant:
            answered.append(result)
        else:
            no_match.append({
                "query_id": query["id"],
                "top1": prediction["ranking"][0],
            })

    if len(answered) != 10 or len(no_match) != 2:
        raise RuntimeError("Query group mismatch")

    macro = {
        name: statistics.mean(
            result[name]
            for result in answered
        )
        for name in ("precision", "recall", "hit")
    }

    latencies = [
        item["seconds"]
        for item in latency_samples
    ]

    if len(latencies) != 60:
        raise RuntimeError(
            "Expected 60 latency samples"
        )

    query_peak = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    image_peak = metadata["peak_rss_bytes"]

    p95 = p95_nearest_rank(latencies)

    gates = {
        "hit_at_5": macro["hit"] >= 0.8,
        "image_encode_time": (
            metadata["image_encode_seconds"] <= 600
        ),
        "query_p95": p95 <= 2.0,
        "peak_rss": (
            max(image_peak, query_peak)
            <= 6 * 1024 ** 3
        ),
    }

    metrics = {
        "k": 5,
        "answered_queries": 10,
        "no_match_queries": 2,
        "macro_answered": macro,
        "per_query": per_query,
        "no_match_observations": no_match,
        "performance": {
            "image_encode_seconds": (
                metadata["image_encode_seconds"]
            ),
            "query_model_load_seconds": load_seconds,
            "warmup_queries": 3,
            "latency_sample_count": 60,
            "latency_samples": latency_samples,
            "p95_method": "nearest_rank",
            "query_p95_seconds": p95,
            "image_process_peak_rss_bytes": image_peak,
            "query_process_peak_rss_bytes": query_peak,
            "max_process_peak_rss_gib": (
                max(image_peak, query_peak)
                / 1024 ** 3
            ),
        },
        "gates": gates,
        "numeric_gates_passed": all(gates.values()),
        "qualitative_review_completed": False,
    }
    # 推理结束后，再次确认输入未变化
    verify_inputs(run)

    rankings = {
        "run_id": run_id,
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": MANIFEST_SHA,
        "model_revision": REVISION,
        "queries": predictions,
    }

    eval_metadata = {
        "run_id": run_id,
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": MANIFEST_SHA,
        "image_cache_metadata_sha256": sha256(
            run / "run_metadata.json"
        ),
        "image_vectors_sha256": sha256(
            run / "image_vectors.npy"
        ),
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "device": "cpu",
        "dtype": "float32",
        "platform": platform.platform(),
        **current_versions,
    }

    report_lines = [
        "# Chinese-CLIP CPU 检索实验",
        "",
        f"- Run：{run_id}",
        f"- 模型 revision：{REVISION}",
        "- 图片：50 张；查询：12 条。",
        "- 有答案查询：10 条；无匹配查询：2 条。",
        "",
        "## 质量",
        "",
        f"- Macro Precision@5：{macro['precision']:.4f}",
        f"- Macro Recall@5：{macro['recall']:.4f}",
        f"- Macro Hit@5：{macro['hit']:.4f}",
        "",
        "## 性能",
        "",
        (
            "- 图片编码时间："
            f"{metadata['image_encode_seconds']:.4f} 秒"
        ),
        f"- 查询 p95：{p95:.4f} 秒。",
        (
            "- 两进程峰值 RSS 较大者："
            f"{max(image_peak, query_peak) / 1024 ** 3:.4f} GiB"
        ),
        "",
        "## 预设门槛",
        "",
    ]

    for name, passed in gates.items():
        report_lines.append(
            f"- {name}：{'通过' if passed else '未通过'}"
        )

    report_lines.extend([
        "",
        "## 逐查询结果",
        "",
        "| 查询 | 相关数 | Precision@5 | Recall@5 | Hit@5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])

    for result in per_query:
        report_lines.append(
            f"| {result['query_id']} "
            f"| {result['relevant_count']} "
            f"| {result['precision']} "
            f"| {result['recall']} "
            f"| {result['hit']} |"
        )

    report_lines.extend([
        "",
        "## 尚待人工分析",
        "",
        "- 逐查询查看 Top-5 图片，解释典型命中与失败。",
        "- 检查多条件查询是否只满足部分条件。",
        "- 无匹配结果只作观察，未建立拒答阈值。",
        "- 本轮为小样本开发基线，不代表整个图库效果。",
        "- 人工标签经 Agent 建议辅助审批，并非盲标。",
        "- 本报告没有自动标记整个 M2 验收通过。",
        "",
    ])

    # 全部文件准备好后，再发布结果目录
    with TemporaryDirectory(
        prefix=".evaluating-",
        dir=run,
    ) as temporary:
        stage = Path(temporary)

        write_json(stage / "rankings.json", rankings)
        write_json(stage / "metrics.json", metrics)
        write_json(
            stage / "evaluation_metadata.json",
            eval_metadata,
        )

        (stage / "report.md").write_text(
            "\n".join(report_lines),
            encoding="utf-8",
        )

        if output.exists():
            raise FileExistsError(output)

        stage.rename(output)

    print("Macro metrics:", macro)
    print("Query p95 seconds:", round(p95, 4))
    print("Numeric gates:", gates)
    print("Saved to:", output)


if __name__ == "__main__":
    main()