
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from photoindex.search_service import SearchService


ROOT = Path(__file__).resolve().parents[1]

RUN = (
    ROOT
    / "local-data/retrieval/runs/chinese-clip-base-cpu-v2"
)

MANIFEST_FILE = (
    ROOT / "local-data/retrieval/manifest.json"
)

RANKINGS_FILE = (
    RUN / "evaluation-v1/rankings.json"
)


def read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def parse_saved_ranking(entry):
    if not isinstance(entry, dict):
        raise ValueError(
            "Ranking entry is "
            f"{type(entry).__name__}; "
            "inspect the actual frozen schema"
        )

    if "asset_id" not in entry or "score" not in entry:
        raise ValueError(
            "Unrecognized saved ranking fields: "
            f"{sorted(entry)}. "
            "Inspect actual record before changing parser."
        )

    asset_id = entry["asset_id"]
    score = entry["score"]

    if (
        type(asset_id) is not int
        or type(score) not in (int, float)
    ):
        raise ValueError(
            "Saved ranking has invalid ID/score types"
        )

    if not math.isfinite(float(score)):
        raise ValueError(
            "Saved ranking contains nonfinite score"
        )

    return asset_id, float(score)


def main():
    manifest = read_json(MANIFEST_FILE)
    rankings = read_json(RANKINGS_FILE)

    expected_manifest_hash = (
        "90ba5129aa1fd4b4ce462dbfd9043088bf235738ab1662edef37523803cdf496"
    )

    assert sha256(MANIFEST_FILE) == expected_manifest_hash

    assert rankings["run_id"] == (
        "chinese-clip-base-cpu-v2"
    )

    assert rankings["dataset_id"] == manifest["dataset_id"]

    assert rankings["manifest_sha256"] == (
        expected_manifest_hash
    )

    assert rankings["model_revision"] == (
        "36e679e65c2a2fead755ae21162091293ad37834"
    )

    queries = manifest["queries"]

    by_id = {
        item["query_id"]: item
        for item in rankings["queries"]
    }

    assert len(queries) == len(by_id) == 12
    assert len({q["id"] for q in queries}) == 12

    print("Loading real SearchService...", flush=True)

    service = SearchService.from_project(ROOT)

    assert service.run_id == rankings["run_id"]
    assert len(service.asset_ids) == 50

    print("Service READY", flush=True)

    total_pairs = 0
    greatest_error = 0.0

    for query in queries:
        query_id = query["id"]

        saved = by_id[query_id]

        assert saved["text"] == query["text"], (
            f"Text mismatch for {query_id}"
        )

        full_ranking = [
            parse_saved_ranking(row)
            for row in saved["ranking"]
        ]

        assert len(full_ranking) == 50
        assert len({
            item[0] for item in full_ranking
        }) == 50

        response = service.search(query["text"])

        assert response.query == query["text"]
        assert response.run_id == rankings["run_id"]
        assert len(response.results) == 5

        actual_ids = [
            result.asset_id
            for result in response.results
        ]

        expected_ids = [
            item[0]
            for item in full_ranking[:5]
        ]

        assert actual_ids == expected_ids, (
            f"{query_id} IDs mismatch: "
            f"expected {expected_ids}, "
            f"got {actual_ids}"
        )

        for result, (
            expected_id,
            expected_score,
        ) in zip(
            response.results,
            full_ranking[:5],
        ):
            assert result.asset_id == expected_id
            assert math.isfinite(result.score)

            assert np.isclose(
                result.score,
                expected_score,
                rtol=1e-4,
                atol=1e-5,
            ), (
                f"{query_id} Asset {result.asset_id} "
                f"score mismatch: "
                f"expected {expected_score:.8f}, "
                f"got {result.score:.8f}"
            )

            greatest_error = max(
                greatest_error,
                abs(result.score - expected_score),
            )

            total_pairs += 1

        print(
            f"PASS {query_id}: {actual_ids}",
            flush=True,
        )

    assert total_pairs == 60

    print(
        "\nPASS: all 12 queries preserve "
        "frozen Top-5 IDs and ordering"
    )
    print(
        "PASS: all 60 scores within "
        "rtol=1e-4 / atol=1e-5"
    )
    print(
        "maximum absolute score difference:",
        greatest_error,
    )
    print("PASS: M3-A frozen ranking regression")


if __name__ == "__main__":
    main()
