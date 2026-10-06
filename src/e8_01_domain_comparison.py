from m02_text_processor import run_preprocessing_experiment
from e4_01_zipf import analyze_frequency


# ============================================================
# EXTENSION E81 — OPTIONAL TWO-DOMAIN COMPARISON
# ============================================================
#
# Purpose:
#     Reuse the SAME preprocessing + Zipf pipeline for two domains.
#     No second IR system is created.
#
# Default comparison condition:
#     C_no_stopwords
#
# Output required by the optional challenge:
#     Documents
#     Tokens
#     Vocabulary
#     Zipf exponent
#     R²
#     Top 10 terms
# ============================================================


def _top_terms(condition_result, top_n=10):
    rows = list(condition_result.get("top_terms", []))
    if len(rows) < top_n:
        frequency = condition_result.get("collection_frequency", {})
        rows = [
            {
                "rank": rank,
                "term": term,
                "collection_frequency": count,
                "document_frequency": condition_result.get(
                    "document_frequency", {}
                ).get(term, 0),
            }
            for rank, (term, count) in enumerate(
                sorted(
                    frequency.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )[:top_n],
                start=1,
            )
        ]
    return rows[:top_n]


def _analyze_domain(processed_documents, label, condition_key, top_n):
    preprocessing = run_preprocessing_experiment(
        processed_documents,
        top_n=max(50, top_n),
    )

    condition = preprocessing.get("conditions", {}).get(condition_key)
    if condition is None:
        raise ValueError(f"Unknown preprocessing condition: {condition_key}")

    zipf = analyze_frequency(
        condition.get("collection_frequency", {}),
        source=f"domain:{label}:{condition_key}",
        top_n=None,
    )

    regression = zipf.get("regression", {})

    return {
        "label": label,
        "condition_key": condition_key,
        "documents": condition.get("document_count", 0),
        "tokens": condition.get("total_tokens", 0),
        "vocabulary": condition.get("unique_terms", 0),
        "average_tokens_per_document": condition.get(
            "average_tokens_per_document", 0.0
        ),
        "zipf_exponent": regression.get("zipf_exponent"),
        "slope": regression.get("slope"),
        "intercept": regression.get("intercept"),
        "r_squared": regression.get("r_squared"),
        "rmse": regression.get("rmse"),
        "best_rank_region": zipf.get("rank_region_analysis", {}).get(
            "best_region_by_r_squared"
        ),
        "top_terms": _top_terms(condition, top_n=top_n),
        "zipf": zipf,
    }


def compare_two_domains(
    processed_documents_a,
    processed_documents_b,
    label_a="Domain A",
    label_b="Domain B",
    condition_key="C_no_stopwords",
    top_n=10,
):
    processed_documents_a = list(processed_documents_a or [])
    processed_documents_b = list(processed_documents_b or [])

    if not processed_documents_a:
        raise ValueError("Domain A has no processed documents.")
    if not processed_documents_b:
        raise ValueError("Domain B has no processed documents.")

    domain_a = _analyze_domain(
        processed_documents_a,
        label=label_a,
        condition_key=condition_key,
        top_n=top_n,
    )
    domain_b = _analyze_domain(
        processed_documents_b,
        label=label_b,
        condition_key=condition_key,
        top_n=top_n,
    )

    comparison_rows = []
    for metric, key in (
        ("Documents", "documents"),
        ("Tokens", "tokens"),
        ("Vocabulary", "vocabulary"),
        ("Average tokens/document", "average_tokens_per_document"),
        ("Zipf exponent", "zipf_exponent"),
        ("R²", "r_squared"),
        ("RMSE", "rmse"),
    ):
        comparison_rows.append(
            {
                "measure": metric,
                label_a: domain_a.get(key),
                label_b: domain_b.get(key),
            }
        )

    return {
        "condition_key": condition_key,
        "domain_a": domain_a,
        "domain_b": domain_b,
        "comparison_rows": comparison_rows,
    }


if __name__ == "__main__":
    print("E81 is a reusable two-domain analyzer. Use it through M08.")
