from pathlib import Path

import numpy as np
from numpy import ndarray


def rank_images(
    asset_ids: list[int],
    image_vectors: ndarray,
    query_vector: ndarray,
) -> list[tuple[int, float]]:

    # 1. 检查 Asset ID
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("Duplicate Asset IDs")

    if any(type(asset_id) is not int for asset_id in asset_ids):
        raise ValueError("Asset IDs must be integers")

    # 2. 检查输入形状
    if image_vectors.ndim != 2:
        raise ValueError("image_vectors must be 2D")

    if query_vector.ndim != 1:
        raise ValueError("query_vector must be 1D")

    if image_vectors.shape[0] != len(asset_ids):
        raise ValueError("Asset/vector row count mismatch")

    if image_vectors.shape[1] != query_vector.shape[0]:
        raise ValueError("Embedding dimensions do not match")

    if image_vectors.shape[1] != 512:
        raise ValueError("Expected 512-dimensional embeddings")

    # 3. 检查数值
    if not np.isfinite(image_vectors).all():
        raise ValueError("Invalid image vectors")

    if not np.isfinite(query_vector).all():
        raise ValueError("Invalid query vector")

    image_norms = np.linalg.norm(image_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)

    if np.any(image_norms == 0) or query_norm == 0:
        raise ValueError("Zero embedding")

    # 要求调用方已经做过归一化
    if not np.allclose(image_norms, 1.0, atol=1e-4):
        raise ValueError("Image vectors are not normalized")

    if not np.isclose(query_norm, 1.0, atol=1e-4):
        raise ValueError("Query vector is not normalized")

    # 4. 矩阵与向量相乘
    scores = image_vectors @ query_vector

    # 5. 转回 Asset ID，稳定排序
    results = [
        (asset_id, float(score))
        for asset_id, score in zip(asset_ids, scores)
    ]

    results.sort(
        key=lambda item: (-item[1], item[0])
    )

    return results

def _normalize_features(features):
    import torch

    if not torch.is_tensor(features):
        raise TypeError("Expected a PyTorch Tensor")

    if features.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")

    features = features.float()

    if not torch.isfinite(features).all():
        raise ValueError("Non-finite features")

    norms = features.norm(p=2, dim=-1, keepdim=True)

    if torch.any(norms == 0):
        raise ValueError("Zero feature vector")

    normalized = features / norms

    result = normalized.cpu().numpy().astype(
        np.float32,
        copy=False,
    )

    if result.shape[1] != 512:
        raise ValueError("Expected 512 dimensions")

    return result

def encode_images(
    paths: list[Path],
    model,
    processor,
    device: str,
) -> ndarray:
    import torch
    from PIL import Image

    if not paths:
        raise ValueError("No image paths")

    model.eval()

    batch_size = 4
    batches = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]

        images = []

        # 读取本批图片
        for path in batch_paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))

        # Processor 将图片转换为 Tensor
        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # 模型推理
        with torch.inference_mode():
            features = model.get_image_features(**inputs)

        # 归一化并转换成 NumPy
        vectors = _normalize_features(features)

        batches.append(vectors)

    return np.concatenate(batches, axis=0)

def encode_texts(
    texts: list[str],
    model,
    processor,
    device: str,
) -> ndarray:
    import torch

    if not texts:
        raise ValueError("No texts")

    if any(not text.strip() for text in texts):
        raise ValueError("Empty text")

    model.eval()

    batch_size = 1
    batches = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        inputs = processor(
            text=batch,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        max_length = model.config.text_config.max_position_embeddings

        if inputs["input_ids"].shape[1] > max_length:
            raise ValueError("Text exceeds model limit")

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            # ChineseCLIPModel disables its text pooler. Transformers 4.57.1
            # get_text_features() nevertheless projects pooler_output=None.
            # Match the CLS projection used by the model's joint forward.
            text_outputs = model.text_model(**inputs, return_dict=True)
            features = model.text_projection(
                text_outputs.last_hidden_state[:, 0, :]
            )

        batches.append(
            _normalize_features(features)
        )

    return np.concatenate(batches, axis=0)
