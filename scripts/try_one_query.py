import json
import numpy as np

from pathlib import Path
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from photoindex.retrieval import encode_texts, rank_images


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "local-data/retrieval"
RUN_DIR = DATA_DIR / "runs/chinese-clip-base-cpu-v1"
MODEL_DIR = ROOT / "local-data/models/chinese-clip-base"


def read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main():
    # 1. 读取之前生成的缓存
    asset_ids = read_json(RUN_DIR / "asset_ids.json")

    image_vectors = np.load(
        RUN_DIR / "image_vectors.npy",
        allow_pickle=False,
    )

    manifest = read_json(DATA_DIR / "manifest.json")

    if asset_ids != manifest["asset_ids"]:
        raise RuntimeError("Asset order mismatch")

    # 2. 选择第一条中文查询
    query = manifest["queries"][0]

    print("Query:", query["text"])

    # 3. 离线加载模型
    processor = ChineseCLIPProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model = ChineseCLIPModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model = model.to("cpu")
    model.eval()

    # 4. 中文文本编码
    text_vectors = encode_texts(
        texts=[query["text"]],
        model=model,
        processor=processor,
        device="cpu",
    )

    assert text_vectors.shape == (1, 512)

    query_vector = text_vectors[0]

    # 5. 计算相似度与排序
    ranking = rank_images(
        asset_ids=asset_ids,
        image_vectors=image_vectors,
        query_vector=query_vector,
    )

    assert len(ranking) == 50

    # 6. 展示前五名
    for position, (asset_id, score) in enumerate(
        ranking[:5],
        start=1,
    ):
        print(
            f"#{position}: Asset {asset_id}, "
            f"score={score:.4f}"
        )


if __name__ == "__main__":
    main()