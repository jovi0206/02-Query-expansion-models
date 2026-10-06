import math


# ============================================================
# EXTENSION E42 — CF / DF / IDF / TF-IDF SUPPORT
# ============================================================
#
# Parent Module:
#     M04 — Indexing
#
# Purpose:
#     Support the Project 2 analysis of:
#         Collection Frequency (CF)
#         Document Frequency (DF)
#         Inverse Document Frequency (IDF)
#         CF vs DF comparison
#         Optional document-level TF-IDF inspection
#
# Teacher formula:
#     IDF(t) = log(N / DF(t))
#
# V3 display convention uses log10 and no smoothing. The assignment
# specifies log(N / DF) without fixing a base; log10 keeps the project
# presentation consistent with the Zipf log-log analysis.
# ============================================================


def compute_idf(total_documents, document_frequency, log_base=10):
    total_documents = int(total_documents or 0)
    document_frequency = int(document_frequency or 0)

    if total_documents <= 0 or document_frequency <= 0:
        return None

    ratio = total_documents / document_frequency

    if log_base in ("e", None):
        return math.log(ratio)
    if log_base == 10:
        return math.log10(ratio)
    if log_base == 2:
        return math.log2(ratio)

    numeric_base = float(log_base)
    if numeric_base <= 0 or numeric_base == 1:
        raise ValueError("log_base must be 'e', 2, 10, or another positive base != 1.")

    return math.log(ratio, numeric_base)


def compute_tf_idf(
    term_frequency,
    total_documents,
    document_frequency,
    log_base=10,
):
    idf = compute_idf(
        total_documents,
        document_frequency,
        log_base=log_base,
    )

    if idf is None:
        return None

    return float(term_frequency or 0) * idf


def extract_from_index(index_data):
    index_data = index_data or {}
    total_documents = int(index_data.get("total_documents", 0))

    cf = {}
    df = {}

    for term, entry in index_data.get("index", {}).items():
        cf[str(term)] = int(entry.get("total_frequency", 0))
        df[str(term)] = int(entry.get("document_frequency", 0))

    return {
        "source": "m04_index",
        "total_documents": total_documents,
        "collection_frequency": cf,
        "document_frequency": df,
    }


def extract_from_condition(condition_result):
    condition_result = condition_result or {}

    return {
        "source": condition_result.get("condition", "e22_condition"),
        "total_documents": int(condition_result.get("document_count", 0)),
        "collection_frequency": {
            str(term): int(value)
            for term, value in condition_result.get(
                "collection_frequency", {}
            ).items()
        },
        "document_frequency": {
            str(term): int(value)
            for term, value in condition_result.get(
                "document_frequency", {}
            ).items()
        },
    }


def build_term_rows(
    total_documents,
    collection_frequency,
    document_frequency,
    terms=None,
    log_base=10,
):
    cf = collection_frequency or {}
    df = document_frequency or {}

    if terms is None:
        terms = sorted(
            set(cf) | set(df),
            key=lambda term: (-int(cf.get(term, 0)), term),
        )
    else:
        terms = [str(term).casefold().strip() for term in terms if str(term).strip()]

    rows = []

    for term in terms:
        term_cf = int(cf.get(term, 0))
        term_df = int(df.get(term, 0))
        idf = compute_idf(
            total_documents,
            term_df,
            log_base=log_base,
        )

        rows.append(
            {
                "term": term,
                "collection_frequency": term_cf,
                "document_frequency": term_df,
                "document_coverage": (
                    term_df / total_documents
                    if total_documents
                    else 0.0
                ),
                # Useful for explaining "high CF but relatively low DF":
                # repeated occurrences concentrated in fewer documents.
                "cf_per_containing_document": (
                    term_cf / term_df
                    if term_df
                    else None
                ),
                "idf": idf,
            }
        )

    return rows


def analyze_term_statistics(
    index_data=None,
    condition_result=None,
    terms=None,
    cf_df_top_n=20,
    idf_top_n=10,
    log_base=10,
):
    """
    Build Project 2 CF/DF/IDF tables.

    Exactly one source should normally be supplied:
        index_data       -> baseline M04 index
        condition_result -> one E22 preprocessing condition
    """
    if condition_result is not None:
        source_data = extract_from_condition(condition_result)
    elif index_data is not None:
        source_data = extract_from_index(index_data)
    else:
        raise ValueError("index_data or condition_result is required.")

    rows = build_term_rows(
        source_data["total_documents"],
        source_data["collection_frequency"],
        source_data["document_frequency"],
        terms=terms,
        log_base=log_base,
    )

    # CF-vs-DF assignment table: default Top 20 by CF.
    cf_df_rows = sorted(
        rows,
        key=lambda row: (
            -row["collection_frequency"],
            -row["document_frequency"],
            row["term"],
        ),
    )[:max(0, int(cf_df_top_n))]

    # IDF table: same high-CF candidates by default, so the relationship
    # between common terms and low IDF is directly visible.
    idf_rows = cf_df_rows[:max(0, int(idf_top_n))]

    return {
        "source": source_data["source"],
        "total_documents": source_data["total_documents"],
        "log_base": log_base,
        "idf_formula": "log10(N / DF)" if log_base == 10 else "log(N / DF)",
        "term_count": len(rows),
        "rows": rows,
        "cf_df_rows": cf_df_rows,
        "idf_rows": idf_rows,
    }


def document_tfidf_from_index(
    index_data,
    document_id,
    terms=None,
    log_base=10,
    normalize_tf=False,
    top_n=None,
):
    """
    Optional inspection helper for document-level TF-IDF.

    TF is raw within-document count by default. If normalize_tf=True,
    TF = count / total_search_words for that document.
    """
    index_data = index_data or {}
    documents = index_data.get("documents", {})

    if document_id not in documents:
        raise ValueError(f"Unknown document_id: {document_id}")

    total_documents = int(index_data.get("total_documents", len(documents)))
    document_word_count = int(
        documents[document_id].get("total_search_words", 0)
    )

    index = index_data.get("index", {})

    if terms is None:
        candidate_terms = list(index)
    else:
        candidate_terms = [
            str(term).casefold().strip()
            for term in terms
            if str(term).strip()
        ]

    rows = []

    for term in candidate_terms:
        entry = index.get(term)
        if not entry:
            continue

        postings = entry.get("postings", {}).get(document_id, [])
        raw_tf = len(postings)
        if raw_tf <= 0:
            continue

        df = int(entry.get("document_frequency", 0))
        idf = compute_idf(total_documents, df, log_base=log_base)

        tf = (
            raw_tf / document_word_count
            if normalize_tf and document_word_count
            else float(raw_tf)
        )

        rows.append(
            {
                "term": term,
                "raw_tf": raw_tf,
                "tf": tf,
                "df": df,
                "idf": idf,
                "tf_idf": (tf * idf if idf is not None else None),
            }
        )

    rows.sort(
        key=lambda row: (
            -(row["tf_idf"] if row["tf_idf"] is not None else -1),
            row["term"],
        )
    )

    if top_n is not None:
        rows = rows[:max(0, int(top_n))]

    return {
        "document_id": document_id,
        "document_word_count": document_word_count,
        "total_documents": total_documents,
        "normalize_tf": bool(normalize_tf),
        "log_base": log_base,
        "rows": rows,
    }


if __name__ == "__main__":
    demo_index = {
        "total_documents": 10,
        "documents": {
            "D1": {"total_search_words": 100},
        },
        "index": {
            "the": {
                "total_frequency": 50,
                "document_frequency": 10,
                "postings": {"D1": [{}] * 5},
            },
            "cancer": {
                "total_frequency": 20,
                "document_frequency": 4,
                "postings": {"D1": [{}] * 2},
            },
        },
    }

    result = analyze_term_statistics(demo_index)
    print("E42 - TERM STATISTICS TEST")
    print(result["cf_df_rows"])
    print(document_tfidf_from_index(demo_index, "D1"))
