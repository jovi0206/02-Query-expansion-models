import math


# ============================================================
# EXTENSION E61 — SIGNIFICANT WORDS / RESOLVING-POWER EXPLORATION
# ============================================================
#
# Classroom concept:
#     very common terms -> low discrimination
#     very rare terms   -> little/reliable evidence
#     middle-frequency terms may have higher resolving power
#
# IMPORTANT:
#     The Project 2 assignment / lecture figure does not provide a
#     numerical formula for Luhn/van Rijsbergen upper/lower cut-offs.
#     V3 therefore DOES NOT invent fixed cut-off ranks.
#
# Instead V3 exposes a transparent exploratory proxy that can be used
# to rank candidate significant terms without claiming that it is the
# professor's resolving-power formula:
#
#     candidate_score = log10(1 + CF) * IDF_log10
#
# Why this is useful as a proxy:
#     - ubiquitous terms have IDF ~= 0 -> low score
#     - very rare terms have small log10(1+CF) -> damped score
#     - terms with repeated use plus cross-document discrimination can
#       rise toward the middle of the frequency distribution
#
# No hard upper/lower cut-off is produced.  The UI labels this explicitly
# as an operational exploratory proxy, not a required-course formula.
# ============================================================


def candidate_resolving_score(collection_frequency, idf):
    cf = max(0, int(collection_frequency or 0))
    if idf is None or cf <= 0:
        return 0.0
    return math.log10(1.0 + cf) * float(idf)


def analyze_significant_words(term_statistics_result, top_n=20):
    stats = term_statistics_result or {}
    source_rows = list(stats.get("rows", []) or [])

    # Ensure deterministic rank by collection frequency, then term.
    ranked = sorted(
        source_rows,
        key=lambda row: (
            -int(row.get("collection_frequency", 0)),
            str(row.get("term", "")),
        ),
    )

    rows = []
    for rank, row in enumerate(ranked, start=1):
        cf = int(row.get("collection_frequency", 0))
        df = int(row.get("document_frequency", 0))
        idf = row.get("idf")
        score = candidate_resolving_score(cf, idf)

        rows.append(
            {
                "rank": rank,
                "term": row.get("term", ""),
                "collection_frequency": cf,
                "document_frequency": df,
                "document_coverage": row.get("document_coverage", 0.0),
                "idf": idf,
                "candidate_resolving_score": score,
            }
        )

    candidates = sorted(
        rows,
        key=lambda row: (
            -float(row.get("candidate_resolving_score", 0.0)),
            row.get("rank", 10**18),
            str(row.get("term", "")),
        ),
    )[:max(0, int(top_n))]

    candidate_ranks = sorted(
        int(row["rank"])
        for row in candidates
        if row.get("rank") is not None
    )

    return {
        "source": stats.get("source", ""),
        "condition_key": stats.get("condition_key"),
        "total_documents": int(stats.get("total_documents", 0)),
        "idf_log_base": stats.get("log_base", 10),
        "method": "exploratory_proxy",
        "formula": "log10(1 + CF) * IDF_log10",
        "cutoff_policy": "none_in_v3",
        "cutoff_note": (
            "No fixed upper/lower cut-off is applied because the course material "
            "does not specify a numerical cut-off algorithm."
        ),
        "term_count": len(rows),
        "rows_by_rank": rows,
        "candidate_terms": candidates,
        "candidate_rank_span": (
            {
                "min_rank": min(candidate_ranks),
                "max_rank": max(candidate_ranks),
            }
            if candidate_ranks
            else None
        ),
    }


def analyze_all_conditions(term_statistics_by_condition, top_n=20):
    output = {}
    for key, stats in (term_statistics_by_condition or {}).items():
        result = analyze_significant_words(stats, top_n=top_n)
        result["condition_key"] = key
        output[key] = result
    return output


if __name__ == "__main__":
    demo = {
        "source": "demo",
        "condition_key": "B_no_punctuation",
        "total_documents": 1000,
        "log_base": 10,
        "rows": [
            {"term": "the", "collection_frequency": 5000, "document_frequency": 980, "idf": math.log10(1000/980)},
            {"term": "glp-1", "collection_frequency": 1500, "document_frequency": 900, "idf": math.log10(1000/900)},
            {"term": "semaglutide", "collection_frequency": 250, "document_frequency": 120, "idf": math.log10(1000/120)},
            {"term": "xylophone", "collection_frequency": 1, "document_frequency": 1, "idf": 3.0},
        ],
    }
    result = analyze_significant_words(demo, top_n=4)
    for row in result["candidate_terms"]:
        print(row)
