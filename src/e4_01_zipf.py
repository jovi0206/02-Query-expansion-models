import math
from collections import Counter


# ============================================================
# EXTENSION E41 — ZIPF / TERM DISTRIBUTION + REGRESSION
# ============================================================
#
# Parent Module:
#     M04 — Indexing
#
# Input:
#     M04 index_data
#
# Optional Input:
#     E21 Porter result
#     E22 preprocessing experiment result
#
# Output:
#     Collection-frequency distribution
#     Zipf rank-frequency table
#     Frequency spectrum
#     Log-log linear regression
#     slope / intercept / Zipf exponent / R² / RMSE
#     High / Middle / Low rank-region regression
#     Porter before / after comparison
#     Preprocessing-condition comparison
# ============================================================


def frequency_from_index(index_data):
    """Extract term -> total_frequency directly from M04 positional index."""
    if not index_data:
        return {}

    return {
        term: int(entry.get("total_frequency", 0))
        for term, entry in index_data.get("index", {}).items()
        if int(entry.get("total_frequency", 0)) > 0
    }


def normalize_frequency_map(frequency):
    """Accept dict / Counter and remove zero-frequency terms."""
    cleaned = {}

    for term, count in (frequency or {}).items():
        count = int(count)
        if count <= 0:
            continue
        cleaned[str(term)] = count

    return cleaned


def build_zipf_distribution(frequency, top_n=None):
    """Return rank-frequency rows with log10(rank) and log10(frequency)."""
    frequency = normalize_frequency_map(frequency)

    sorted_terms = sorted(
        frequency.items(),
        key=lambda item: (-item[1], item[0]),
    )

    if top_n is not None:
        sorted_terms = sorted_terms[:max(0, int(top_n))]

    total_frequency = sum(frequency.values())
    rows = []

    for rank, (term, count) in enumerate(sorted_terms, start=1):
        rows.append(
            {
                "rank": rank,
                "term": term,
                "frequency": count,
                "probability": (
                    count / total_frequency
                    if total_frequency
                    else 0.0
                ),
                "rank_frequency": rank * count,
                "log_rank": math.log10(rank),
                "log_frequency": math.log10(count),
            }
        )

    return rows


def build_frequency_spectrum(frequency):
    """Return frequency -> number of terms having that frequency."""
    frequency = normalize_frequency_map(frequency)
    spectrum = Counter(frequency.values())

    return [
        {
            "frequency": count,
            "term_count": spectrum[count],
        }
        for count in sorted(spectrum)
    ]


def summarize_frequency(frequency):
    frequency = normalize_frequency_map(frequency)
    total_terms = sum(frequency.values())
    vocabulary_size = len(frequency)
    singleton_terms = sum(1 for count in frequency.values() if count == 1)
    most_common = sorted(
        frequency.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return {
        "total_terms": total_terms,
        "vocabulary_size": vocabulary_size,
        "singleton_terms": singleton_terms,
        "singleton_ratio": (
            singleton_terms / vocabulary_size
            if vocabulary_size
            else 0.0
        ),
        "most_common_term": most_common[0][0] if most_common else "",
        "most_common_frequency": most_common[0][1] if most_common else 0,
    }


# ============================================================
# LOG-LOG LINEAR REGRESSION
# ============================================================

def fit_log_log_regression(zipf_rows):
    """
    Fit:
        log10(CF) = intercept + slope * log10(rank)

    Project 2 reports:
        slope
        intercept
        Zipf exponent = -slope
        R²
        RMSE

    RMSE is measured in log10-frequency units.
    """
    rows = [
        row
        for row in (zipf_rows or [])
        if row.get("rank", 0) > 0 and row.get("frequency", 0) > 0
    ]

    n = len(rows)
    if n < 2:
        return {
            "valid": False,
            "n": n,
            "slope": None,
            "intercept": None,
            "zipf_exponent": None,
            "r_squared": None,
            "rmse": None,
            "rank_start": rows[0]["rank"] if rows else None,
            "rank_end": rows[-1]["rank"] if rows else None,
        }

    x = [float(row["log_rank"]) for row in rows]
    y = [float(row["log_frequency"]) for row in rows]

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    sxx = sum((value - mean_x) ** 2 for value in x)
    if sxx == 0:
        return {
            "valid": False,
            "n": n,
            "slope": None,
            "intercept": None,
            "zipf_exponent": None,
            "r_squared": None,
            "rmse": None,
            "rank_start": rows[0]["rank"],
            "rank_end": rows[-1]["rank"],
        }

    sxy = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x, y))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    predicted = [
        intercept + slope * xv
        for xv in x
    ]

    residuals = [
        yv - prediction
        for yv, prediction in zip(y, predicted)
    ]

    sse = sum(residual ** 2 for residual in residuals)
    sst = sum((yv - mean_y) ** 2 for yv in y)

    r_squared = (
        1.0 - (sse / sst)
        if sst > 0
        else None
    )

    rmse = math.sqrt(sse / n)

    return {
        "valid": True,
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "zipf_exponent": -slope,
        "r_squared": r_squared,
        "rmse": rmse,
        "rank_start": rows[0]["rank"],
        "rank_end": rows[-1]["rank"],
        "log_base": 10,
        "equation": "log10(CF) = intercept + slope * log10(rank)",
    }


def split_rank_regions(zipf_rows):
    """
    Split the rank list into three approximately equal rank ranges.

    This gives a reproducible operational definition for the assignment's
    High-frequency / Middle-frequency / Low-frequency comparison.
    """
    rows = list(zipf_rows or [])
    n = len(rows)

    if n == 0:
        return {
            "high": [],
            "middle": [],
            "low": [],
        }

    first_end = max(1, math.ceil(n / 3))
    second_end = max(first_end, math.ceil(2 * n / 3))

    return {
        "high": rows[:first_end],
        "middle": rows[first_end:second_end],
        "low": rows[second_end:],
    }


def analyze_rank_regions(zipf_rows):
    regions = split_rank_regions(zipf_rows)
    output = {}

    labels = {
        "high": "High-frequency terms",
        "middle": "Middle-frequency terms",
        "low": "Low-frequency terms",
    }

    for key, rows in regions.items():
        regression = fit_log_log_regression(rows)
        regression["region"] = key
        regression["label"] = labels[key]
        output[key] = regression

    valid_regions = [
        item
        for item in output.values()
        if item.get("valid") and item.get("r_squared") is not None
    ]

    best_region = None
    if valid_regions:
        best_region = max(
            valid_regions,
            key=lambda item: (
                item["r_squared"],
                -item["rmse"],
            ),
        )["region"]

    return {
        "method": "Three approximately equal rank ranges",
        "regions": output,
        "best_region_by_r_squared": best_region,
    }


def _build_analysis(frequency, source, top_n=None):
    frequency = normalize_frequency_map(frequency)

    # Regression must use the complete distribution. top_n is display-only.
    full_zipf = build_zipf_distribution(frequency, top_n=None)
    display_zipf = (
        full_zipf
        if top_n is None
        else full_zipf[:max(0, int(top_n))]
    )

    return {
        "source": source,
        "summary": summarize_frequency(frequency),
        "frequency": frequency,
        "zipf": display_zipf,
        "zipf_row_count": len(full_zipf),
        "frequency_spectrum": build_frequency_spectrum(frequency),
        "regression": fit_log_log_regression(full_zipf),
        "rank_region_analysis": analyze_rank_regions(full_zipf),
    }


def analyze_index_zipf(index_data, top_n=None):
    """Main E41 interface for M04 index_data."""
    return _build_analysis(
        frequency_from_index(index_data),
        source="m04_index",
        top_n=top_n,
    )


def analyze_frequency(frequency, source="frequency_map", top_n=None):
    """Analyze any collection-frequency map."""
    return _build_analysis(
        frequency,
        source=source,
        top_n=top_n,
    )


def compare_porter_result(porter_result, top_n=None):
    """Compare E21 original and Porter-stemmed distributions."""
    if not porter_result:
        raise ValueError("porter_result is required.")

    original = analyze_frequency(
        porter_result.get("original_frequency", {}),
        source="original_tokens",
        top_n=top_n,
    )
    stemmed = analyze_frequency(
        porter_result.get("stemmed_frequency", {}),
        source="porter_stemmed_tokens",
        top_n=top_n,
    )

    original_vocab = original["summary"]["vocabulary_size"]
    stemmed_vocab = stemmed["summary"]["vocabulary_size"]
    vocabulary_reduction = original_vocab - stemmed_vocab

    return {
        "original": original,
        "stemmed": stemmed,
        "comparison": {
            "document_count": porter_result.get("document_count", 0),
            "token_count": porter_result.get("token_count", 0),
            "original_vocab_size": original_vocab,
            "stemmed_vocab_size": stemmed_vocab,
            "vocabulary_reduction": vocabulary_reduction,
            "vocabulary_reduction_ratio": (
                vocabulary_reduction / original_vocab
                if original_vocab
                else 0.0
            ),
            "original_singletons": original["summary"]["singleton_terms"],
            "stemmed_singletons": stemmed["summary"]["singleton_terms"],
            "original_zipf_exponent": original["regression"]["zipf_exponent"],
            "stemmed_zipf_exponent": stemmed["regression"]["zipf_exponent"],
            "original_r_squared": original["regression"]["r_squared"],
            "stemmed_r_squared": stemmed["regression"]["r_squared"],
            "original_rmse": original["regression"]["rmse"],
            "stemmed_rmse": stemmed["regression"]["rmse"],
        },
    }


def analyze_preprocessing_experiment(preprocessing_result, top_n=None):
    """
    Run Zipf + regression for every E22 preprocessing condition.
    """
    if not preprocessing_result:
        raise ValueError("preprocessing_result is required.")

    conditions = preprocessing_result.get("conditions", {})
    condition_order = preprocessing_result.get(
        "condition_order",
        list(conditions),
    )

    analyses = {}
    comparison = []

    for condition_key in condition_order:
        condition = conditions.get(condition_key)
        if not condition:
            continue

        analysis = analyze_frequency(
            condition.get("collection_frequency", {}),
            source=condition_key,
            top_n=top_n,
        )
        analyses[condition_key] = analysis

        regression = analysis["regression"]
        summary = analysis["summary"]

        comparison.append(
            {
                "condition": condition_key,
                "label": condition.get("label", condition_key),
                "documents": condition.get("document_count", 0),
                "total_tokens": summary["total_terms"],
                "vocabulary_size": summary["vocabulary_size"],
                "zipf_exponent": regression.get("zipf_exponent"),
                "slope": regression.get("slope"),
                "intercept": regression.get("intercept"),
                "r_squared": regression.get("r_squared"),
                "rmse": regression.get("rmse"),
                "best_rank_region": analysis[
                    "rank_region_analysis"
                ].get("best_region_by_r_squared"),
            }
        )

    # Additional Page 08 Porter comparisons.  These are deliberately kept
    # separate from the formal A-D comparison above.
    porter_demo_source = preprocessing_result.get("porter_demo", {}) or {}
    porter_demo_analyses = {
        "comparison_order": list(porter_demo_source.get("comparison_order", [])),
        "algorithm": porter_demo_source.get("algorithm", "E21 Porter stemming"),
        "formal_conditions_unchanged": True,
        "comparisons": {},
    }

    for before_key in porter_demo_analyses["comparison_order"]:
        demo_entry = (porter_demo_source.get("comparisons", {}) or {}).get(before_key) or {}
        before_condition = conditions.get(before_key) or {}
        before_analysis = analyses.get(before_key)
        if not before_condition or not before_analysis:
            continue

        after_condition_key = demo_entry.get("after_condition_key")
        formal_equivalent = demo_entry.get("formal_equivalent")

        if after_condition_key:
            after_condition = conditions.get(after_condition_key) or {}
            after_analysis = analyses.get(after_condition_key)
        else:
            after_condition = demo_entry.get("after_result") or {}
            after_analysis = (
                analyze_frequency(
                    after_condition.get("collection_frequency", {}),
                    source=f"{before_key}_plus_porter",
                    top_n=top_n,
                )
                if after_condition
                else None
            )

        if not after_condition or not after_analysis:
            continue

        before_reg = before_analysis.get("regression", {})
        after_reg = after_analysis.get("regression", {})
        before_vocab = int(before_condition.get("unique_terms", 0) or 0)
        after_vocab = int(after_condition.get("unique_terms", 0) or 0)
        reduction = before_vocab - after_vocab

        porter_demo_analyses["comparisons"][before_key] = {
            "before_condition_key": before_key,
            "after_condition_key": after_condition_key,
            "formal_equivalent": formal_equivalent,
            "before": before_analysis,
            "after": after_analysis,
            "comparison": {
                "documents": int(before_condition.get("document_count", 0) or 0),
                "tokens_before": int(before_condition.get("total_tokens", 0) or 0),
                "tokens_after": int(after_condition.get("total_tokens", 0) or 0),
                "vocabulary_before": before_vocab,
                "vocabulary_after": after_vocab,
                "vocabulary_reduction": reduction,
                "vocabulary_reduction_ratio": (
                    reduction / before_vocab
                    if before_vocab
                    else 0.0
                ),
                "zipf_k_before": before_reg.get("zipf_exponent"),
                "zipf_k_after": after_reg.get("zipf_exponent"),
                "r_squared_before": before_reg.get("r_squared"),
                "r_squared_after": after_reg.get("r_squared"),
                "rmse_before": before_reg.get("rmse"),
                "rmse_after": after_reg.get("rmse"),
            },
        }

    return {
        "source": "e22_preprocessing_experiment",
        "document_count": preprocessing_result.get("document_count", 0),
        "condition_order": condition_order,
        "conditions": analyses,
        "comparison": comparison,
        "porter_demo": porter_demo_analyses,
    }


if __name__ == "__main__":
    demo_frequency = {
        "the": 1000,
        "of": 500,
        "and": 333,
        "data": 250,
        "model": 200,
        "cancer": 167,
        "rare": 10,
        "veryrare": 5,
    }

    result = analyze_frequency(demo_frequency, source="demo")
    print("E41 - ZIPF + REGRESSION TEST")
    print(result["regression"])
    print(result["rank_region_analysis"])
