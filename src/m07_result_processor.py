# ============================================================
# MODULE 07 — RESULT PROCESSOR
# ============================================================
#
# Input:
#     M06 retrieval_data
#     M03 positioned_documents
#
# Process:
#     1. Add context to raw hits
#     2. Remove duplicate / overlapping related hits
#     3. Sort exact / related hits
#     4. Aggregate result counts
#     5. Group results by document
#     6. Produce final Search Result contract
#
# Output:
#     search_result
#
# M07 answers:
#     "How should the retrieval hits be organized for use?"
#
# M07 does NOT:
#     Parse documents              -> M01
#     Process text                 -> M02
#     Map positions                -> M03
#     Build index                  -> M04
#     Interpret query              -> M05
#     Retrieve matches             -> M06
#     Control application flow     -> M08
#     Render UI                    -> M09
# ============================================================


def resolve_document_id(document):
    return (
        document.get("pmcid")
        or document.get("pmid")
        or document["filename"]
    )


def build_document_lookup(positioned_documents):
    return {
        resolve_document_id(document): document
        for document in positioned_documents
    }


def find_segment(document, segment_id):
    for segment in document.get("segments", []):
        if segment.get("segment_id") == segment_id:
            return segment
    return None


def find_sentence_for_char(segment, char_start):
    for sentence in segment.get("sentences", []):
        if (
            sentence.get("global_char_start", -1)
            <= char_start
            < sentence.get("global_char_end", -1)
        ):
            return sentence

    for sentence in segment.get("sentences", []):
        if (
            sentence.get("global_char_start", -1)
            <= char_start
            <= sentence.get("global_char_end", -1)
        ):
            return sentence

    return None


def get_context(document, segment, char_start, char_end, radius=120):
    sentence = find_sentence_for_char(segment, char_start)

    if sentence:
        return sentence.get("text", "")

    text = document.get("search_text", "")
    left = max(0, char_start - radius)
    right = min(len(text), char_end + radius)

    return text[left:right].replace("\n", " ")


def add_context_to_hit(hit, document_lookup):
    item = dict(hit)
    document = document_lookup.get(item.get("document_id"))

    if not document:
        item["context"] = ""
        return item

    segment = find_segment(document, item.get("segment_id"))

    if segment is None:
        item["context"] = ""
        return item

    item["context"] = get_context(
        document=document,
        segment=segment,
        char_start=item.get("char_start", 0),
        char_end=item.get("char_end", 0),
    )

    return item


def ranges_overlap(first, second):
    return (
        first["char_start"] < second["char_end"]
        and second["char_start"] < first["char_end"]
    )


def unique_hits(hits):
    filtered = []
    seen = set()

    for item in hits:
        key = (
            item.get("document_id"),
            item.get("segment_id"),
            item.get("char_start"),
            item.get("char_end"),
            item.get("match_method"),
        )

        if key in seen:
            continue

        seen.add(key)
        filtered.append(item)

    return filtered


def remove_exact_overlaps(exact_results, related_results):
    filtered = []

    for related in related_results:
        duplicate = any(
            related["document_id"] == exact["document_id"]
            and related["segment_id"] == exact["segment_id"]
            and ranges_overlap(related, exact)
            for exact in exact_results
        )

        if duplicate:
            continue

        filtered.append(related)

    return filtered


def sort_exact_results(exact_results):
    return sorted(
        exact_results,
        key=lambda item: (
            str(item.get("document_id", "")),
            item.get("char_start", 0),
            item.get("char_end", 0),
        ),
    )


def sort_related_results(related_results):
    return sorted(
        related_results,
        key=lambda item: (
            -float(item.get("similarity", 0.0)),
            str(item.get("document_id", "")),
            item.get("char_start", 0),
            item.get("char_end", 0),
        ),
    )


def group_results_by_document(all_results):
    grouped = {}

    for item in all_results:
        document_id = item["document_id"]
        grouped.setdefault(document_id, []).append(item)

    return grouped


def build_document_results(grouped_results):
    document_results = []

    for document_id, items in grouped_results.items():
        exact_count = sum(
            1
            for item in items
            if item.get("match_type") == "exact"
        )

        related_count = len(items) - exact_count

        best_similarity = max(
            (
                float(item.get("similarity", 0.0))
                for item in items
            ),
            default=0.0,
        )

        title = items[0].get("title", "") if items else ""

        document_results.append(
            {
                "document_id": document_id,
                "title": title,
                "exact_matches": exact_count,
                "related_matches": related_count,
                "total_matches": len(items),
                "best_similarity": round(best_similarity, 4),
                "results": items,
            }
        )

    document_results.sort(
        key=lambda item: (
            -item["exact_matches"],
            -item["total_matches"],
            str(item["document_id"]),
        )
    )

    for rank, item in enumerate(document_results, start=1):
        item["document_rank"] = rank

    return document_results


def empty_search_result(retrieval_data=None, total_documents=0):
    retrieval_data = retrieval_data or {}
    query_plan = retrieval_data.get("query_plan", {})

    return {
        "query": retrieval_data.get("query", ""),
        "query_type": retrieval_data.get("query_type", "empty"),
        "requested_query_type": query_plan.get(
            "requested_query_type",
            "empty",
        ),
        "retrieval_mode": retrieval_data.get(
            "retrieval_mode",
            "empty",
        ),
        "fallback_used": retrieval_data.get("fallback_used", False),
        "total_documents": total_documents,
        "documents_found": 0,
        "exact_matches": 0,
        "related_matches": 0,
        "total_matches": 0,
        "exact_results": [],
        "related_results": [],
        "all_results": [],
        "grouped_results": {},
        "document_results": [],
        "result_stage": "processed_results",
    }


def process_results(retrieval_data, positioned_documents):
    if not retrieval_data:
        return empty_search_result(
            total_documents=len(positioned_documents or [])
        )

    total_documents = len(positioned_documents or [])

    raw_exact = retrieval_data.get("exact_results", [])
    raw_related = retrieval_data.get("related_results", [])

    if not raw_exact and not raw_related:
        return empty_search_result(
            retrieval_data=retrieval_data,
            total_documents=total_documents,
        )

    document_lookup = build_document_lookup(positioned_documents)

    exact_results = [
        add_context_to_hit(item, document_lookup)
        for item in unique_hits(raw_exact)
    ]

    related_results = [
        add_context_to_hit(item, document_lookup)
        for item in unique_hits(raw_related)
    ]

    related_results = remove_exact_overlaps(
        exact_results,
        related_results,
    )

    exact_results = sort_exact_results(exact_results)
    related_results = sort_related_results(related_results)

    all_results = exact_results + related_results

    for rank, item in enumerate(all_results, start=1):
        item["result_rank"] = rank

    grouped_results = group_results_by_document(all_results)
    document_results = build_document_results(grouped_results)

    query_plan = retrieval_data.get("query_plan", {})

    return {
        "query": retrieval_data.get("query", ""),
        "query_type": retrieval_data.get("query_type", "empty"),
        "requested_query_type": query_plan.get(
            "requested_query_type",
            "empty",
        ),
        "retrieval_mode": retrieval_data.get(
            "retrieval_mode",
            "empty",
        ),
        "fallback_used": retrieval_data.get("fallback_used", False),
        "total_documents": total_documents,
        "documents_found": len(grouped_results),
        "exact_matches": len(exact_results),
        "related_matches": len(related_results),
        "total_matches": len(all_results),
        "exact_results": exact_results,
        "related_results": related_results,
        "all_results": all_results,
        "grouped_results": grouped_results,
        "document_results": document_results,
        "result_stage": "processed_results",
    }


if __name__ == "__main__":
    positioned_documents = [
        {
            "filename": "demo.xml",
            "pmid": "DEMO001",
            "title": "Demo Article",
            "search_text": "Physical activity improves health.",
            "segments": [
                {
                    "segment_id": 0,
                    "field": "abstract",
                    "text": "Physical activity improves health.",
                    "sentences": [
                        {
                            "text": "Physical activity improves health.",
                            "global_char_start": 0,
                            "global_char_end": 34,
                        }
                    ],
                }
            ],
        }
    ]

    retrieval_data = {
        "query": "physical activity",
        "query_type": "phrase",
        "retrieval_mode": "positional",
        "fallback_used": False,
        "query_plan": {
            "requested_query_type": "auto",
        },
        "exact_results": [
            {
                "document_id": "DEMO001",
                "title": "Demo Article",
                "segment_id": 0,
                "field": "abstract",
                "match_type": "exact",
                "match_method": "exact_phrase",
                "matched_text": "Physical activity",
                "similarity": 1.0,
                "word_position": 0,
                "sentence_position": 0,
                "char_start": 0,
                "char_end": 17,
            }
        ],
        "related_results": [
            {
                "document_id": "DEMO001",
                "title": "Demo Article",
                "segment_id": 0,
                "field": "abstract",
                "match_type": "related",
                "match_method": "proximity_N2",
                "matched_text": "Physical activity",
                "similarity": 1.0,
                "word_position": 0,
                "sentence_position": 0,
                "char_start": 0,
                "char_end": 17,
            }
        ],
    }

    result = process_results(
        retrieval_data=retrieval_data,
        positioned_documents=positioned_documents,
    )

    print("=" * 80)
    print("MODULE 07 - RESULT PROCESSOR TEST")
    print("=" * 80)
    print("Documents :", result["documents_found"])
    print("Exact     :", result["exact_matches"])
    print("Related   :", result["related_matches"])
    print("Total     :", result["total_matches"])
    print("Context   :", result["exact_results"][0]["context"])
