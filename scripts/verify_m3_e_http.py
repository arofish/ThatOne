
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]

RUN = (
    ROOT
    / "local-data/retrieval/runs"
    / "chinese-clip-base-cpu-v2"
)

MANIFEST_PATH = (
    ROOT / "local-data/retrieval/manifest.json"
)

RANKINGS_PATH = (
    RUN / "evaluation-v1/rankings.json"
)

METADATA_PATH = RUN / "run_metadata.json"

TOKEN_PATH = (
    Path.home() / ".config/thatone/dev-token"
)

RUN_ID = "chinese-clip-base-cpu-v2"

BASE_URL = "http://127.0.0.1:8765"


def read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main() -> None:
    manifest = read_json(MANIFEST_PATH)
    saved = read_json(RANKINGS_PATH)
    metadata = read_json(METADATA_PATH)

    token = TOKEN_PATH.read_text(
        encoding="utf-8"
    ).strip()

    if len(token) < 32:
        raise RuntimeError("Missing valid local token")

    queries = manifest["queries"]
    saved_queries = {
        row["query_id"]: row
        for row in saved["queries"]
    }

    assert len(queries) == 12
    assert len(saved_queries) == 12
    assert saved["run_id"] == RUN_ID
    assert saved["dataset_id"] == manifest["dataset_id"]

    manifest_hash = hashlib.sha256(
        MANIFEST_PATH.read_bytes()
    ).hexdigest()

    assert manifest_hash == (
        "90ba5129aa1fd4b4ce462dbfd9043088bf235738ab1662edef37523803cdf496"
    )
    assert saved["manifest_sha256"] == manifest_hash

    headers = {
        "Authorization": f"Bearer {token}",
    }

    checked = 0
    largest_error = 0.0

    with httpx.Client(
        base_url=BASE_URL,
        timeout=10.0,
        trust_env=False,
        headers=headers,
    ) as client:

        health = client.get("/health")
        health.raise_for_status()

        assert health.json() == {
            "status": "ready",
            "run_id": RUN_ID,
            "asset_count": 50,
        }

        print("PASS: authenticated /health")

        for query in queries:
            query_id = query["id"]
            text = query["text"]

            expected = saved_queries[query_id]

            assert expected["text"] == text
            assert len(expected["ranking"]) == 50

            response = client.post(
                "/search",
                json={"query": text},
            )

            response.raise_for_status()
            result = response.json()

            assert result["query"] == text
            assert result["run_id"] == RUN_ID
            assert len(result["results"]) == 5

            actual_ids = [
                item["asset_id"]
                for item in result["results"]
            ]

            expected_ids = [
                item["asset_id"]
                for item in expected["ranking"][:5]
            ]

            assert actual_ids == expected_ids, (
                f"{query_id}: ID/ranking mismatch"
            )

            for actual, frozen in zip(
                result["results"],
                expected["ranking"][:5],
            ):
                score = float(actual["score"])
                reference = float(frozen["score"])

                assert math.isfinite(score)

                assert math.isclose(
                    score,
                    reference,
                    rel_tol=1e-4,
                    abs_tol=1e-5,
                ), f"{query_id}: score mismatch"

                assert actual["thumbnail_url"] == (
                    "/assets/"
                    f"{actual['asset_id']}/thumbnail"
                )

                largest_error = max(
                    largest_error,
                    abs(score - reference),
                )

                checked += 1

            print(
                "PASS",
                query_id,
                actual_ids,
                flush=True,
            )

        assert checked == 60

        # 验证实际 HTTP 返回的缩略图字节身份。
        preview = client.get(
            "/assets/19/thumbnail"
        )
        preview.raise_for_status()

        expected_preview = next(
            row["sha256"]
            for row in metadata["inputs"]
            if row["asset_id"] == 19
        )

        actual_preview = hashlib.sha256(
            preview.content
        ).hexdigest()

        assert actual_preview == expected_preview
        assert preview.headers["content-type"].startswith(
            "image/png"
        )

        print("PASS: thumbnail bytes match frozen v2")

    print("\n=== M3-E HTTP REGRESSION ===")
    print("Queries:", len(queries))
    print("Top-5 score comparisons:", checked)
    print("Largest absolute score error:", largest_error)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
