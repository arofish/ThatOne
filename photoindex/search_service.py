
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from photoindex.cache_validation import load_verified_v2_cache
from photoindex.retrieval import encode_texts, rank_images


# ---------- 错误类型 ----------

class SearchValidationError(ValueError):
    """The query is invalid for the text encoder."""


class ThumbnailNotFoundError(FileNotFoundError):
    """The requested thumbnail is unknown or missing."""


class PreviewChangedError(RuntimeError):
    """The thumbnail differs from the verified cache."""


class ThumbnailReadError(OSError):
    """The thumbnail cannot be read."""


# ---------- 搜索结果 ----------

@dataclass(frozen=True)
class SearchResult:
    asset_id: int
    score: float
    relative_path: str
    original_path: str
    original_status: str
    thumbnail_url: str


@dataclass(frozen=True)
class SearchResponse:
    query: str
    run_id: str
    results: list[SearchResult]


# ---------- 原片当前状态 ----------

def check_original_status(path_string: str) -> str:
    path = Path(path_string)

    try:
        if not path.is_file():
            return "missing"

        with path.open("rb"):
            pass

        return "available"

    except FileNotFoundError:
        return "missing"

    except (PermissionError, OSError):
        return "unreadable"


# ---------- 搜索服务 ----------

class SearchService:

    def __init__(
        self,
        model,
        processor,
        asset_ids: tuple[int, ...],
        image_vectors: np.ndarray,
        asset_paths: dict[int, tuple[str, str]],
        thumbnails: dict[int, tuple[Path, str]],
        run_id: str,
        text_token_limit: int,
    ) -> None:

        if type(text_token_limit) is not int or text_token_limit < 3:
            raise ValueError("Invalid text token limit")

        if (
            image_vectors.ndim != 2
            or image_vectors.shape != (len(asset_ids), 512)
        ):
            raise ValueError(
                "Image vector cache must have shape (N, 512)"
            )

        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("Duplicate Asset ID")

        if (
            set(asset_ids) != set(asset_paths)
            or set(asset_ids) != set(thumbnails)
        ):
            raise ValueError(
                "Cache mappings must include exactly the frozen Asset IDs"
            )

        self.model = model
        self.processor = processor
        self.asset_ids = tuple(asset_ids)
        self.image_vectors = image_vectors
        self.asset_paths = dict(asset_paths)
        self.thumbnails = dict(thumbnails)
        self.run_id = run_id
        self.text_token_limit = text_token_limit

        self._inference_lock = Lock()

    # ---------- 正式项目初始化 ----------

    @classmethod
    def from_project(
        cls,
        project_root: Path,
    ) -> SearchService:

        # 先验证缓存，再加载数百 MB 的模型。
        verified = load_verified_v2_cache(project_root)

        # 延迟导入：仅调用 from_project 时才加载 AI 框架。
        import sys
        import torch
        import transformers
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        if sys.version_info[:2] != (3, 13):
            raise RuntimeError("M3 requires Python 3.13")

        actual_versions = (
            np.__version__,
            torch.__version__.split("+", 1)[0],
            transformers.__version__,
        )

        if actual_versions != ("2.5.4", "2.11.0", "4.57.1"):
            raise RuntimeError(
                "AI dependency versions differ from the accepted "
                f"M2 baseline: {actual_versions}"
            )

        model = ChineseCLIPModel.from_pretrained(
            str(verified.model_dir),
            local_files_only=True,
        )

        model = model.to(
            device="cpu",
            dtype=torch.float32,
        )
        model.eval()

        processor = ChineseCLIPProcessor.from_pretrained(
            str(verified.model_dir),
            use_fast=False,
            local_files_only=True,
        )

        # 模型的位置编码规定了实际输入长度边界。
        text_config = getattr(
            model.config,
            "text_config",
            None,
        )

        position_limit = getattr(
            text_config,
            "max_position_embeddings",
            None,
        )

        if (
            type(position_limit) is not int
            or position_limit < 3
        ):
            raise RuntimeError(
                "Model text position limit is missing or invalid"
            )

        tokenizer = getattr(
            processor,
            "tokenizer",
            None,
        )

        if tokenizer is None:
            raise RuntimeError("Processor has no tokenizer")

        tokenizer_limit = getattr(
            tokenizer,
            "model_max_length",
            None,
        )

        if (
            type(tokenizer_limit) is int
            and 3 <= tokenizer_limit < 1_000_000
        ):
            text_token_limit = min(
                position_limit,
                tokenizer_limit,
            )
        else:
            text_token_limit = position_limit

        return cls(
            model=model,
            processor=processor,
            asset_ids=verified.asset_ids,
            image_vectors=verified.image_vectors,
            asset_paths=verified.asset_paths,
            thumbnails=verified.thumbnails,
            run_id=verified.run_id,
            text_token_limit=text_token_limit,
        )

    # ---------- 查询合法性检查 ----------

    def _validate_query(self, query: str) -> None:

        if not isinstance(query, str):
            raise SearchValidationError(
                "Query must be a string"
            )

        if not query.strip():
            raise SearchValidationError(
                "Query must not be blank"
            )

        if len(query) > 1000:
            raise SearchValidationError(
                "Query exceeds 1000 characters"
            )

        # 不改写原文，不做静默截断。
        encoded = self.processor.tokenizer(
            query,
            add_special_tokens=True,
            truncation=False,
        )

        token_ids = encoded["input_ids"]

        if (
            not isinstance(token_ids, list)
            or not all(type(i) is int for i in token_ids)
        ):
            raise RuntimeError(
                "Unexpected tokenizer output"
            )

        if len(token_ids) > self.text_token_limit:
            raise SearchValidationError(
                f"Query has {len(token_ids)} tokens; "
                f"maximum is {self.text_token_limit}"
            )

    # ---------- 中文语义搜索 ----------

    def search(self, query: str) -> SearchResponse:

        with self._inference_lock:

            self._validate_query(query)

            text_vectors = encode_texts(
                [query],
                self.model,
                self.processor,
                "cpu",
            )

            if (
                not isinstance(text_vectors, np.ndarray)
                or text_vectors.shape != (1, 512)
            ):
                raise RuntimeError(
                    "Text encoder returned an invalid vector shape"
                )

            if not bool(np.isfinite(text_vectors).all()):
                raise RuntimeError(
                    "Text encoder returned nonfinite values"
                )

            ranking = rank_images(
                self.asset_ids,
                self.image_vectors,
                text_vectors[0],
            )

        # 先取模型排名，再检查文件状态。
        top5 = ranking[:5]

        results = []

        for asset_id, score in top5:

            relative_path, original_path = (
                self.asset_paths[asset_id]
            )

            results.append(
                SearchResult(
                    asset_id=asset_id,
                    score=float(score),
                    relative_path=relative_path,
                    original_path=original_path,
                    original_status=check_original_status(
                        original_path
                    ),
                    thumbnail_url=(
                        f"/assets/{asset_id}/thumbnail"
                    ),
                )
            )

        return SearchResponse(
            query=query,
            run_id=self.run_id,
            results=results,
        )

    # ---------- 已冻结缩略图读取 ----------

    def thumbnail_bytes(self, asset_id: int) -> bytes:

        if (
            type(asset_id) is not int
            or asset_id not in self.thumbnails
        ):
            raise ThumbnailNotFoundError(
                "Unknown thumbnail Asset ID"
            )

        path, expected_digest = self.thumbnails[asset_id]

        try:
            data = path.read_bytes()

        except FileNotFoundError as exc:
            raise ThumbnailNotFoundError(
                "Thumbnail file is missing"
            ) from exc

        except OSError as exc:
            raise ThumbnailReadError(
                "Cannot read thumbnail"
            ) from exc

        actual_digest = hashlib.sha256(data).hexdigest()

        if actual_digest != expected_digest:
            raise PreviewChangedError(
                "Thumbnail differs from the frozen v2 input"
            )

        return data
