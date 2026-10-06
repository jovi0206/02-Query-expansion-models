import re
from collections import Counter

from m05_query_processor import (
    lexical_base,
    normalize_surface,
)


# ============================================================
# MODULE 06 — RETRIEVAL ENGINE
# ============================================================
#
# Input:
#     M05 query_plan
#     M03 positioned_documents
#     M04 index_data
#
# Process:
#     1. Exact Word Retrieval
#     2. Related Word Retrieval
#     3. Exact Phrase Retrieval
#     4. Proximity Phrase Retrieval
#     5. Exact Sentence Retrieval
#     6. Approximate Sentence Retrieval
#     7. Literal Substring Retrieval
#     8. Auto Literal Fallback
#
# Output:
#     retrieval_data
#
# M06 only answers:
#     "Where are the matches?"
#
# M06 does NOT:
#     Decide the query type          -> M05
#     Perform spelling correction    -> E5x
#     Rank / deduplicate results     -> M07
#     Display UI                     -> M09
#
# Note:
#     M06 intentionally returns RAW retrieval hits.
#     Result sorting / deduplication / aggregation will be
#     handled by M07 Result Processor.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

# Related phrase:
# all query words must exist inside one sentence.
# Up to N extra words may appear inside the matched span.
PHRASE_PROXIMITY_N = 2

# Approximate sentence:
# query words must remain in order.
SENTENCE_EXTRA_WORD_RATIO = 0.30
SENTENCE_MIN_EXTRA_WORDS = 2

# Prefix relation:
# avoid treating very short strings such as "an" as roots.
RELATED_PREFIX_MIN_LENGTH = 4


# ============================================================
# 1. DOCUMENT HELPERS
# ============================================================

def resolve_document_id(document):
    return (
        document.get("pmcid")
        or document.get("pmid")
        or document["filename"]
    )


def build_document_lookup(
    positioned_documents,
):
    return {
        resolve_document_id(
            document
        ):
            document

        for document
        in positioned_documents
    }


def find_segment(
    document,
    segment_id,
):
    for segment in document[
        "segments"
    ]:
        if (
            segment["segment_id"]
            == segment_id
        ):
            return segment

    return None


def find_sentence_for_char(
    segment,
    char_start,
):
    for sentence in segment[
        "sentences"
    ]:
        if (
            sentence[
                "global_char_start"
            ]
            <= char_start
            < sentence[
                "global_char_end"
            ]
        ):
            return sentence

    # Fallback:
    # useful for a literal match beginning at punctuation/space.
    for sentence in segment[
        "sentences"
    ]:
        if (
            sentence[
                "global_char_start"
            ]
            <= char_start
            <= sentence[
                "global_char_end"
            ]
        ):
            return sentence

    return None


def get_sentence_words(
    segment,
    sentence,
):
    start = sentence[
        "global_char_start"
    ]

    end = sentence[
        "global_char_end"
    ]

    return [
        word

        for word
        in segment["words"]

        if (
            start
            <= word[
                "global_char_start"
            ]
            < end
        )
    ]


def find_word_position_for_range(
    segment,
    char_start,
    char_end,
):
    # First preference:
    # query begins inside this word.
    for word in segment[
        "words"
    ]:
        if (
            word[
                "global_char_start"
            ]
            <= char_start
            < word[
                "global_char_end"
            ]
        ):
            return word[
                "global_word_index"
            ]

    # Second preference:
    # query begins in punctuation/space.
    # Use the first overlapping word.
    for word in segment[
        "words"
    ]:
        if (
            word[
                "global_char_start"
            ]
            < char_end

            and

            char_start
            < word[
                "global_char_end"
            ]
        ):
            return word[
                "global_word_index"
            ]

    return None


# ============================================================
# 2. STANDARD RAW HIT FORMAT
# ============================================================

def make_hit(
    document,
    segment,
    match_type,
    match_method,
    char_start,
    char_end,
    word_position,
    sentence_position,
    similarity=1.0,
):
    """
    Standard M06 raw retrieval hit.

    M07 will later:
        deduplicate
        sort
        rank
        aggregate
        build result summaries / context
    """

    return {
        "document_id":
            resolve_document_id(
                document
            ),

        "title":
            document.get(
                "title",
                "",
            ),

        "segment_id":
            segment[
                "segment_id"
            ],

        "field":
            segment[
                "field"
            ],

        "match_type":
            match_type,

        "match_method":
            match_method,

        "matched_text":
            document[
                "search_text"
            ][
                char_start:
                char_end
            ],

        "similarity":
            round(
                similarity,
                4,
            ),

        "word_position":
            word_position,

        "sentence_position":
            sentence_position,

        "char_start":
            char_start,

        "char_end":
            char_end,
    }


# ============================================================
# 3. RELATED WORD RULE
# ============================================================

def related_word_relation(
    query_term,
    candidate_term,
):
    query_surface = normalize_surface(
        query_term
    )

    candidate_surface = normalize_surface(
        candidate_term
    )

    if (
        not query_surface
        or not candidate_surface
    ):
        return None

    # Exact is handled separately.
    if query_surface == candidate_surface:
        return None

    query_base = lexical_base(
        query_term
    )

    candidate_base = lexical_base(
        candidate_term
    )

    # Singular / plural / possessive
    if query_base == candidate_base:
        return {
            "method":
                "lexical_variant",

            "similarity":
                1.0,
        }

    # Prefix-related form
    if (
        len(query_base)
        >= RELATED_PREFIX_MIN_LENGTH
        and candidate_base.startswith(
            query_base
        )
    ):
        similarity = (
            len(query_base)
            /
            len(candidate_base)
        )

        return {
            "method":
                "prefix_related",

            "similarity":
                round(
                    similarity,
                    4,
                ),
        }

    if (
        len(candidate_base)
        >= RELATED_PREFIX_MIN_LENGTH
        and query_base.startswith(
            candidate_base
        )
    ):
        similarity = (
            len(candidate_base)
            /
            len(query_base)
        )

        return {
            "method":
                "prefix_related",

            "similarity":
                round(
                    similarity,
                    4,
                ),
        }

    return None


# ============================================================
# 4. EXACT WORD
# ============================================================

def search_exact_word(
    index_data,
    document_lookup,
    query_plan,
):
    if not query_plan[
        "normalized_terms"
    ]:
        return []

    query_surface = query_plan[
        "normalized_terms"
    ][0]

    results = []

    for (
        indexed_term,
        entry,
    ) in index_data[
        "index"
    ].items():

        if (
            normalize_surface(
                indexed_term
            )
            != query_surface
        ):
            continue

        for (
            document_id,
            postings,
        ) in entry[
            "postings"
        ].items():

            document = document_lookup[
                document_id
            ]

            for posting in postings:

                segment = find_segment(
                    document,
                    posting[
                        "segment_id"
                    ],
                )

                if segment is None:
                    continue

                results.append(
                    make_hit(
                        document=
                            document,

                        segment=
                            segment,

                        match_type=
                            "exact",

                        match_method=
                            "exact_word",

                        char_start=
                            posting[
                                "global_char_start"
                            ],

                        char_end=
                            posting[
                                "global_char_end"
                            ],

                        word_position=
                            posting[
                                "global_word_index"
                            ],

                        sentence_position=
                            posting[
                                "sentence_index"
                            ],
                    )
                )

    return results


# ============================================================
# 5. RELATED WORD
# ============================================================

def search_related_word(
    index_data,
    document_lookup,
    query_plan,
):
    if not query_plan[
        "tokens"
    ]:
        return []

    query_term = query_plan[
        "tokens"
    ][0]

    results = []

    for (
        indexed_term,
        entry,
    ) in index_data[
        "index"
    ].items():

        relation = (
            related_word_relation(
                query_term,
                indexed_term,
            )
        )

        if relation is None:
            continue

        for (
            document_id,
            postings,
        ) in entry[
            "postings"
        ].items():

            document = document_lookup[
                document_id
            ]

            for posting in postings:

                segment = find_segment(
                    document,
                    posting[
                        "segment_id"
                    ],
                )

                if segment is None:
                    continue

                results.append(
                    make_hit(
                        document=
                            document,

                        segment=
                            segment,

                        match_type=
                            "related",

                        match_method=
                            relation[
                                "method"
                            ],

                        char_start=
                            posting[
                                "global_char_start"
                            ],

                        char_end=
                            posting[
                                "global_char_end"
                            ],

                        word_position=
                            posting[
                                "global_word_index"
                            ],

                        sentence_position=
                            posting[
                                "sentence_index"
                            ],

                        similarity=
                            relation[
                                "similarity"
                            ],
                    )
                )

    return results


# ============================================================
# 6. EXACT PHRASE
# ============================================================
#
# Query words must be:
#     same order
#     adjacent
#     inside the same sentence
# ============================================================

def search_exact_phrase(
    positioned_documents,
    query_plan,
):
    query_terms = query_plan[
        "normalized_terms"
    ]

    if len(query_terms) < 2:
        return []

    results = []

    query_length = len(
        query_terms
    )

    for document in positioned_documents:

        for segment in document[
            "segments"
        ]:

            for sentence in segment[
                "sentences"
            ]:

                words = get_sentence_words(
                    segment,
                    sentence,
                )

                if (
                    len(words)
                    < query_length
                ):
                    continue

                terms = [
                    normalize_surface(
                        word["text"]
                    )

                    for word
                    in words
                ]

                for start in range(
                    len(words)
                    - query_length
                    + 1
                ):

                    end = (
                        start
                        + query_length
                    )

                    if (
                        terms[
                            start:end
                        ]
                        != query_terms
                    ):
                        continue

                    first_word = words[
                        start
                    ]

                    last_word = words[
                        end - 1
                    ]

                    results.append(
                        make_hit(
                            document=
                                document,

                            segment=
                                segment,

                            match_type=
                                "exact",

                            match_method=
                                "exact_phrase",

                            char_start=
                                first_word[
                                    "global_char_start"
                                ],

                            char_end=
                                last_word[
                                    "global_char_end"
                                ],

                            word_position=
                                first_word[
                                    "global_word_index"
                                ],

                            sentence_position=
                                sentence[
                                    "global_sentence_index"
                                ],
                        )
                    )

    return results


# ============================================================
# 7. PROXIMITY HELPERS
# ============================================================

def counter_contains(
    window_counter,
    query_counter,
):
    return all(
        window_counter[
            term
        ]
        >= required

        for (
            term,
            required,
        )
        in query_counter.items()
    )


def find_proximity_windows(
    words,
    query_terms,
    max_extra_words,
):
    if (
        not words
        or not query_terms
    ):
        return []

    terms = [
        lexical_base(
            word["text"]
        )

        for word
        in words
    ]

    query_counter = Counter(
        query_terms
    )

    query_length = len(
        query_terms
    )

    maximum_window = (
        query_length
        + max_extra_words
    )

    windows = []
    seen = set()

    for start in range(
        len(words)
    ):

        if (
            terms[start]
            not in query_counter
        ):
            continue

        max_end = min(
            len(words),
            start + maximum_window,
        )

        for end in range(
            start + query_length,
            max_end + 1,
        ):

            window_terms = (
                terms[
                    start:end
                ]
            )

            if (
                window_terms[-1]
                not in query_counter
            ):
                continue

            if not counter_contains(
                Counter(
                    window_terms
                ),
                query_counter,
            ):
                continue

            key = (
                start,
                end,
            )

            if key not in seen:

                seen.add(
                    key
                )

                windows.append(
                    {
                        "start":
                            start,

                        "end":
                            end,

                        "extra_words":
                            (
                                end
                                - start
                                - query_length
                            ),
                    }
                )

            # Keep the shortest valid window for this start.
            break

    return windows


# ============================================================
# 8. RELATED PHRASE
# ============================================================

def search_related_phrase(
    positioned_documents,
    query_plan,
    proximity_n=
        PHRASE_PROXIMITY_N,
):
    query_terms = query_plan[
        "lexical_terms"
    ]

    if len(query_terms) < 2:
        return []

    results = []

    for document in positioned_documents:

        for segment in document[
            "segments"
        ]:

            for sentence in segment[
                "sentences"
            ]:

                words = get_sentence_words(
                    segment,
                    sentence,
                )

                windows = (
                    find_proximity_windows(
                        words,
                        query_terms,
                        max_extra_words=
                            proximity_n,
                    )
                )

                for window in windows:

                    first_word = words[
                        window["start"]
                    ]

                    last_word = words[
                        window["end"] - 1
                    ]

                    span_length = (
                        window["end"]
                        - window["start"]
                    )

                    similarity = (
                        len(query_terms)
                        /
                        span_length
                    )

                    results.append(
                        make_hit(
                            document=
                                document,

                            segment=
                                segment,

                            match_type=
                                "related",

                            match_method=
                                f"proximity_N{proximity_n}",

                            char_start=
                                first_word[
                                    "global_char_start"
                                ],

                            char_end=
                                last_word[
                                    "global_char_end"
                                ],

                            word_position=
                                first_word[
                                    "global_word_index"
                                ],

                            sentence_position=
                                sentence[
                                    "global_sentence_index"
                                ],

                            similarity=
                                similarity,
                        )
                    )

    return results


# ============================================================
# 9. EXACT SENTENCE
# ============================================================

def search_exact_sentence(
    positioned_documents,
    query_plan,
):
    query_terms = query_plan[
        "normalized_terms"
    ]

    if len(query_terms) < 2:
        return []

    results = []

    for document in positioned_documents:

        for segment in document[
            "segments"
        ]:

            for sentence in segment[
                "sentences"
            ]:

                words = get_sentence_words(
                    segment,
                    sentence,
                )

                candidate_terms = [
                    normalize_surface(
                        word["text"]
                    )

                    for word
                    in words
                ]

                if (
                    candidate_terms
                    != query_terms
                ):
                    continue

                results.append(
                    make_hit(
                        document=
                            document,

                        segment=
                            segment,

                        match_type=
                            "exact",

                        match_method=
                            "exact_sentence",

                        char_start=
                            sentence[
                                "global_char_start"
                            ],

                        char_end=
                            sentence[
                                "global_char_end"
                            ],

                        word_position=
                            (
                                words[0][
                                    "global_word_index"
                                ]
                                if words
                                else None
                            ),

                        sentence_position=
                            sentence[
                                "global_sentence_index"
                            ],
                    )
                )

    return results


# ============================================================
# 10. ORDERED PROXIMITY FOR SENTENCE
# ============================================================

def find_ordered_windows(
    words,
    query_terms,
    max_extra_words,
):
    terms = [
        lexical_base(
            word["text"]
        )

        for word
        in words
    ]

    if (
        not terms
        or not query_terms
    ):
        return []

    windows = []
    seen = set()

    first_term = (
        query_terms[0]
    )

    for start in range(
        len(terms)
    ):

        if (
            terms[start]
            != first_term
        ):
            continue

        cursor = (
            start + 1
        )

        matched_positions = [
            start
        ]

        success = True

        for query_term in query_terms[
            1:
        ]:

            found = False

            while (
                cursor
                < len(terms)
            ):

                if (
                    terms[cursor]
                    == query_term
                ):

                    matched_positions.append(
                        cursor
                    )

                    cursor += 1
                    found = True
                    break

                cursor += 1

            if not found:
                success = False
                break

        if not success:
            continue

        end = (
            matched_positions[-1]
            + 1
        )

        span_length = (
            end
            - start
        )

        extra_words = (
            span_length
            - len(query_terms)
        )

        if (
            extra_words
            > max_extra_words
        ):
            continue

        key = (
            start,
            end,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        windows.append(
            {
                "start":
                    start,

                "end":
                    end,

                "extra_words":
                    extra_words,
            }
        )

    return windows


# ============================================================
# 11. APPROXIMATE SENTENCE
# ============================================================

def search_approximate_sentence(
    positioned_documents,
    query_plan,
    extra_word_ratio=
        SENTENCE_EXTRA_WORD_RATIO,
):
    query_terms = query_plan[
        "lexical_terms"
    ]

    if len(query_terms) < 2:
        return []

    max_extra_words = max(
        SENTENCE_MIN_EXTRA_WORDS,
        round(
            len(query_terms)
            * extra_word_ratio
        ),
    )

    results = []

    for document in positioned_documents:

        for segment in document[
            "segments"
        ]:

            for sentence in segment[
                "sentences"
            ]:

                words = get_sentence_words(
                    segment,
                    sentence,
                )

                windows = (
                    find_ordered_windows(
                        words,
                        query_terms,
                        max_extra_words=
                            max_extra_words,
                    )
                )

                for window in windows:

                    first_word = words[
                        window["start"]
                    ]

                    last_word = words[
                        window["end"] - 1
                    ]

                    span_length = (
                        window["end"]
                        - window["start"]
                    )

                    similarity = (
                        len(query_terms)
                        /
                        span_length
                    )

                    results.append(
                        make_hit(
                            document=
                                document,

                            segment=
                                segment,

                            match_type=
                                "related",

                            match_method=
                                "ordered_proximity",

                            char_start=
                                first_word[
                                    "global_char_start"
                                ],

                            char_end=
                                last_word[
                                    "global_char_end"
                                ],

                            word_position=
                                first_word[
                                    "global_word_index"
                                ],

                            sentence_position=
                                sentence[
                                    "global_sentence_index"
                                ],

                            similarity=
                                similarity,
                        )
                    )

    return results


# ============================================================
# 12. LITERAL SUBSTRING SEARCH
# ============================================================

def build_literal_pattern(
    query,
):
    query = query.strip()

    if not query:
        return None

    # Make copied line breaks / multiple spaces tolerant.
    parts = [
        part

        for part
        in re.split(
            r"\s+",
            query,
        )

        if part
    ]

    if not parts:
        return None

    pattern_text = (
        r"\s+"
        .join(
            re.escape(
                part
            )

            for part
            in parts
        )
    )

    return re.compile(
        pattern_text,
        re.IGNORECASE,
    )


def search_literal_text(
    positioned_documents,
    query_plan,
):
    pattern = build_literal_pattern(
        query_plan[
            "query"
        ]
    )

    if pattern is None:
        return []

    results = []

    for document in positioned_documents:

        for segment in document[
            "segments"
        ]:

            segment_text = segment[
                "text"
            ]

            segment_start = segment[
                "global_char_start"
            ]

            for match in pattern.finditer(
                segment_text
            ):

                char_start = (
                    segment_start
                    + match.start()
                )

                char_end = (
                    segment_start
                    + match.end()
                )

                sentence = (
                    find_sentence_for_char(
                        segment,
                        char_start,
                    )
                )

                sentence_position = (
                    sentence[
                        "global_sentence_index"
                    ]
                    if sentence
                    else None
                )

                word_position = (
                    find_word_position_for_range(
                        segment,
                        char_start,
                        char_end,
                    )
                )

                results.append(
                    make_hit(
                        document=
                            document,

                        segment=
                            segment,

                        match_type=
                            "exact",

                        match_method=
                            "literal_substring",

                        char_start=
                            char_start,

                        char_end=
                            char_end,

                        word_position=
                            word_position,

                        sentence_position=
                            sentence_position,

                        similarity=
                            1.0,
                    )
                )

    return results


# ============================================================
# 13. EMPTY RETRIEVAL
# ============================================================

def empty_retrieval_data(
    query_plan,
):
    return {
        "query_plan":
            query_plan,

        "query":
            query_plan.get(
                "query",
                "",
            ),

        "query_type":
            query_plan.get(
                "query_type",
                "empty",
            ),

        "retrieval_mode":
            "empty",

        "fallback_used":
            False,

        "exact_results":
            [],

        "related_results":
            [],

        "raw_total_matches":
            0,

        "result_stage":
            "raw_retrieval",
    }


# ============================================================
# 14. MAIN RETRIEVAL ENGINE
# ============================================================

def retrieve(
    index_data,
    positioned_documents,
    query_plan,
    phrase_proximity_n=
        PHRASE_PROXIMITY_N,
):
    """
    Execute retrieval according to the Query Plan created by M05.

    Important:
        This function intentionally does NOT:
            deduplicate results
            rank results
            aggregate per-document counts
            build final UI summaries

        Those are M07 responsibilities.
    """

    if (
        not query_plan
        or query_plan.get(
            "is_empty",
            True,
        )
    ):
        return empty_retrieval_data(
            query_plan
            or {}
        )

    query_type = query_plan[
        "query_type"
    ]

    auto_requested = query_plan[
        "auto_requested"
    ]

    document_lookup = (
        build_document_lookup(
            positioned_documents
        )
    )

    retrieval_mode = (
        "positional"
    )

    fallback_used = False

    # --------------------------------------------------------
    # EXPLICIT LITERAL
    # --------------------------------------------------------

    if query_type == "literal":

        exact_results = (
            search_literal_text(
                positioned_documents,
                query_plan,
            )
        )

        related_results = []

        retrieval_mode = (
            "literal"
        )

    # --------------------------------------------------------
    # WORD
    # --------------------------------------------------------

    elif query_type == "word":

        exact_results = (
            search_exact_word(
                index_data,
                document_lookup,
                query_plan,
            )
        )

        related_results = (
            search_related_word(
                index_data,
                document_lookup,
                query_plan,
            )
        )

    # --------------------------------------------------------
    # PHRASE
    # --------------------------------------------------------

    elif query_type == "phrase":

        exact_results = (
            search_exact_phrase(
                positioned_documents,
                query_plan,
            )
        )

        related_results = (
            search_related_phrase(
                positioned_documents,
                query_plan,
                proximity_n=
                    phrase_proximity_n,
            )
        )

    # --------------------------------------------------------
    # SENTENCE
    # --------------------------------------------------------

    elif query_type == "sentence":

        exact_results = (
            search_exact_sentence(
                positioned_documents,
                query_plan,
            )
        )

        related_results = (
            search_approximate_sentence(
                positioned_documents,
                query_plan,
            )
        )

    else:
        raise ValueError(
            "M06 received unsupported query_type: "
            f"{query_type}"
        )

    # --------------------------------------------------------
    # AUTO LITERAL FALLBACK
    # --------------------------------------------------------
    #
    # Preserve Project 1 behavior:
    # if Auto mode finds nothing through normal retrieval,
    # try browser-like literal substring matching.
    # --------------------------------------------------------

    if (
        auto_requested
        and not exact_results
        and not related_results
    ):

        literal_results = (
            search_literal_text(
                positioned_documents,
                query_plan,
            )
        )

        if literal_results:

            exact_results = (
                literal_results
            )

            related_results = []

            retrieval_mode = (
                "literal_fallback"
            )

            fallback_used = True

    return {
        "query_plan":
            query_plan,

        "query":
            query_plan[
                "query"
            ],

        "query_type":
            (
                "literal"
                if fallback_used
                else query_type
            ),

        "retrieval_mode":
            retrieval_mode,

        "fallback_used":
            fallback_used,

        "exact_results":
            exact_results,

        "related_results":
            related_results,

        "raw_total_matches":
            (
                len(exact_results)
                + len(related_results)
            ),

        "result_stage":
            "raw_retrieval",
    }


# ============================================================
# 15. TEST HELPERS
# ============================================================

def print_retrieval_data(
    retrieval_data,
    limit=5,
):
    print()
    print("=" * 80)
    print(
        f"QUERY          : "
        f"{retrieval_data['query']}"
    )
    print(
        f"QUERY TYPE     : "
        f"{retrieval_data['query_type']}"
    )
    print(
        f"MODE           : "
        f"{retrieval_data['retrieval_mode']}"
    )
    print(
        f"FALLBACK       : "
        f"{retrieval_data['fallback_used']}"
    )
    print(
        f"RAW MATCHES    : "
        f"{retrieval_data['raw_total_matches']}"
    )

    print()
    print("EXACT")

    for item in retrieval_data[
        "exact_results"
    ][:limit]:
        print(
            f"[{item['document_id']}] "
            f"{item['match_method']} "
            f"{item['field']} "
            f"Char={item['char_start']}:"
            f"{item['char_end']} "
            f"{item['matched_text']}"
        )

    print()
    print("RELATED")

    for item in retrieval_data[
        "related_results"
    ][:limit]:
        print(
            f"[{item['document_id']}] "
            f"{item['match_method']} "
            f"Similarity={item['similarity']:.2f} "
            f"{item['field']} "
            f"Char={item['char_start']}:"
            f"{item['char_end']} "
            f"{item['matched_text']}"
        )


# ============================================================
# 16. SIMPLE SYNTHETIC TEST
# ============================================================
#
# This test does not depend on PubMed / XML.
# It only verifies the M05 -> M06 interface.
# ============================================================

if __name__ == "__main__":

    from m05_query_processor import (
        process_query,
    )

    positioned_documents = [
        {
            "filename":
                "demo.xml",

            "pmcid":
                "",

            "pmid":
                "DEMO001",

            "title":
                "Demo",

            "search_text":
                (
                    "Physical activity improves health.\n"
                    "Cancer-related outcomes were studied."
                ),

            "segments":
                [
                    {
                        "segment_id":
                            0,

                        "field":
                            "abstract",

                        "text":
                            "Physical activity improves health.",

                        "global_char_start":
                            0,

                        "global_char_end":
                            34,

                        "words":
                            [
                                {
                                    "text": "Physical",
                                    "global_word_index": 0,
                                    "global_char_start": 0,
                                    "global_char_end": 8,
                                },
                                {
                                    "text": "activity",
                                    "global_word_index": 1,
                                    "global_char_start": 9,
                                    "global_char_end": 17,
                                },
                                {
                                    "text": "improves",
                                    "global_word_index": 2,
                                    "global_char_start": 18,
                                    "global_char_end": 26,
                                },
                                {
                                    "text": "health",
                                    "global_word_index": 3,
                                    "global_char_start": 27,
                                    "global_char_end": 33,
                                },
                            ],

                        "sentences":
                            [
                                {
                                    "text":
                                        "Physical activity improves health.",

                                    "global_sentence_index":
                                        0,

                                    "global_char_start":
                                        0,

                                    "global_char_end":
                                        34,
                                }
                            ],
                    },
                    {
                        "segment_id":
                            1,

                        "field":
                            "abstract",

                        "text":
                            "Cancer-related outcomes were studied.",

                        "global_char_start":
                            35,

                        "global_char_end":
                            72,

                        "words":
                            [
                                {
                                    "text": "Cancer-related",
                                    "global_word_index": 4,
                                    "global_char_start": 35,
                                    "global_char_end": 49,
                                },
                                {
                                    "text": "outcomes",
                                    "global_word_index": 5,
                                    "global_char_start": 50,
                                    "global_char_end": 58,
                                },
                                {
                                    "text": "were",
                                    "global_word_index": 6,
                                    "global_char_start": 59,
                                    "global_char_end": 63,
                                },
                                {
                                    "text": "studied",
                                    "global_word_index": 7,
                                    "global_char_start": 64,
                                    "global_char_end": 71,
                                },
                            ],

                        "sentences":
                            [
                                {
                                    "text":
                                        "Cancer-related outcomes were studied.",

                                    "global_sentence_index":
                                        1,

                                    "global_char_start":
                                        35,

                                    "global_char_end":
                                        72,
                                }
                            ],
                    },
                ],
        }
    ]

    index_data = {
        "index": {
            "physical": {
                "document_frequency": 1,
                "total_frequency": 1,
                "postings": {
                    "DEMO001": [
                        {
                            "document_id": "DEMO001",
                            "segment_id": 0,
                            "field": "abstract",
                            "text": "Physical",
                            "word_index": 0,
                            "global_word_index": 0,
                            "char_start": 0,
                            "char_end": 8,
                            "global_char_start": 0,
                            "global_char_end": 8,
                            "sentence_index": 0,
                        }
                    ]
                },
            },
            "activity": {
                "document_frequency": 1,
                "total_frequency": 1,
                "postings": {
                    "DEMO001": [
                        {
                            "document_id": "DEMO001",
                            "segment_id": 0,
                            "field": "abstract",
                            "text": "activity",
                            "word_index": 1,
                            "global_word_index": 1,
                            "char_start": 9,
                            "char_end": 17,
                            "global_char_start": 9,
                            "global_char_end": 17,
                            "sentence_index": 0,
                        }
                    ]
                },
            },
        }
    }

    for query, query_type in [
        ("physical", "word"),
        ("physical activity", "phrase"),
        ("hysical act", "literal"),
    ]:

        plan = process_query(
            query=query,
            query_type=query_type,
        )

        data = retrieve(
            index_data=
                index_data,

            positioned_documents=
                positioned_documents,

            query_plan=
                plan,
        )

        print_retrieval_data(
            data
        )
