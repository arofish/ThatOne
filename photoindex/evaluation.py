def evaluate_at_k(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k: int = 5,
) -> dict[str, float | None]:
    """Evaluate a ranked image retrieval result at K."""

    # 1. 验证 k 必须是正整数
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")

    # 2. 验证完整排序列表没有重复 ID
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked_ids contains duplicates")

    # 3. 只取前 k 个结果
    top_k = ranked_ids[:k]

    # 4. 统计命中数量
    hits = sum(
        asset_id in relevant_ids
        for asset_id in top_k
    )

    # 5. Precision：固定分母 k
    precision = hits / k

    # 6. 无匹配查询采用特殊约定
    if not relevant_ids:
        return {
            "precision": 0.0,
            "recall": None,
            "hit": None,
        }

    # 7. Recall：命中数 / 所有相关图片数量
    recall = hits / len(relevant_ids)

    # 8. Hit：至少命中一个则为 1.0
    hit = 1.0 if hits > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "hit": hit,
    }