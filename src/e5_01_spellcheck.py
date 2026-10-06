from collections import Counter

from m02_text_processor import (
    tokenize_words,
)


# ============================================================
# EXTENSION E51 — EDIT DISTANCE / SPELLING CORRECTION
# ============================================================
#
# Parent Module:
#     M05 — Query Processing
#
# Input:
#     Query
#     M04 index_data / vocabulary
#
# Process:
#     Dynamic Programming Levenshtein Edit Distance
#
# Operations:
#     Insert
#     Delete
#     Substitute
#
# Output:
#     Candidate corrections
#     Corrected Query
#
# Important:
#     E51 does NOT retrieve documents.
#
#     E51:
#         misspelled query
#             ->
#         corrected query
#
#     M05 / M06:
#         corrected query
#             ->
#         retrieval
# ============================================================


# ============================================================
# 1. NORMALIZATION
# ============================================================

def normalize_term(
    term,
):
    if not term:
        return ""

    return (
        str(term)
        .casefold()
        .replace(
            "’",
            "'",
        )
        .strip()
    )


# ============================================================
# 2. FULL DP MATRIX
# ============================================================

def edit_distance_matrix(
    source,
    target,
):
    """
    Build the complete dynamic-programming matrix.

    This is useful for:
        teaching
        debugging
        demo/report screenshots

    matrix[i][j]
        = edit distance between
          source[:i] and target[:j]
    """

    source = normalize_term(
        source
    )

    target = normalize_term(
        target
    )

    rows = len(
        source
    ) + 1

    cols = len(
        target
    ) + 1

    matrix = [
        [
            0
            for _ in range(
                cols
            )
        ]

        for _ in range(
            rows
        )
    ]

    for row in range(
        rows
    ):
        matrix[
            row
        ][0] = row

    for col in range(
        cols
    ):
        matrix[0][
            col
        ] = col

    for row in range(
        1,
        rows,
    ):

        for col in range(
            1,
            cols,
        ):

            substitution_cost = (
                0
                if source[
                    row - 1
                ]
                == target[
                    col - 1
                ]
                else 1
            )

            matrix[
                row
            ][
                col
            ] = min(
                # Delete
                matrix[
                    row - 1
                ][
                    col
                ]
                + 1,

                # Insert
                matrix[
                    row
                ][
                    col - 1
                ]
                + 1,

                # Substitute
                matrix[
                    row - 1
                ][
                    col - 1
                ]
                + substitution_cost,
            )

    return matrix


# ============================================================
# 3. EDIT DISTANCE
# ============================================================

def edit_distance(
    source,
    target,
    max_distance=None,
):
    """
    Memory-efficient Levenshtein Edit Distance.

    Optional max_distance:
        allows early exit when a candidate is already too far.
    """

    source = normalize_term(
        source
    )

    target = normalize_term(
        target
    )

    if source == target:
        return 0

    if not source:
        return len(
            target
        )

    if not target:
        return len(
            source
        )

    if (
        max_distance is not None
        and abs(
            len(source)
            - len(target)
        )
        > max_distance
    ):
        return (
            max_distance
            + 1
        )

    # Keep target as the shorter row when possible.
    if len(source) < len(target):
        source, target = (
            target,
            source,
        )

    previous = list(
        range(
            len(target)
            + 1
        )
    )

    for row_index, source_char in enumerate(
        source,
        start=1,
    ):

        current = [
            row_index
        ]

        row_minimum = row_index

        for col_index, target_char in enumerate(
            target,
            start=1,
        ):

            insertion = (
                current[
                    col_index - 1
                ]
                + 1
            )

            deletion = (
                previous[
                    col_index
                ]
                + 1
            )

            substitution = (
                previous[
                    col_index - 1
                ]
                + (
                    0
                    if source_char
                    == target_char
                    else 1
                )
            )

            value = min(
                insertion,
                deletion,
                substitution,
            )

            current.append(
                value
            )

            row_minimum = min(
                row_minimum,
                value,
            )

        if (
            max_distance is not None
            and row_minimum
            > max_distance
        ):
            return (
                max_distance
                + 1
            )

        previous = current

    return previous[-1]


# ============================================================
# 4. VOCABULARY FROM M04
# ============================================================

def vocabulary_from_index(
    index_data,
):
    """
    Return:
        normalized_term -> total_frequency
    """

    vocabulary = Counter()

    for term, entry in (
        (index_data or {})
        .get(
            "index",
            {},
        )
        .items()
    ):

        normalized = (
            normalize_term(
                term
            )
        )

        if not normalized:
            continue

        vocabulary[
            normalized
        ] += int(
            entry.get(
                "total_frequency",
                0,
            )
        )

    return dict(
        vocabulary
    )


# ============================================================
# 5. CANDIDATE SEARCH
# ============================================================

def find_candidates(
    word,
    vocabulary,
    max_distance=2,
    top_n=5,
):
    """
    Candidate ranking:
        1. Smaller edit distance
        2. Higher corpus frequency
        3. Smaller length difference
        4. Alphabetical stable tie-breaker
    """

    word = normalize_term(
        word
    )

    if not word:
        return []

    if isinstance(
        vocabulary,
        dict,
    ):
        frequency_map = vocabulary
    else:
        frequency_map = {
            normalize_term(
                term
            ): 0

            for term
            in vocabulary
        }

    candidates = []

    for candidate, frequency in (
        frequency_map.items()
    ):

        candidate = normalize_term(
            candidate
        )

        if not candidate:
            continue

        length_difference = abs(
            len(word)
            - len(candidate)
        )

        if (
            length_difference
            > max_distance
        ):
            continue

        distance = edit_distance(
            word,
            candidate,
            max_distance=
                max_distance,
        )

        if (
            distance
            > max_distance
        ):
            continue

        candidates.append(
            {
                "term":
                    candidate,

                "distance":
                    distance,

                "frequency":
                    int(
                        frequency
                    ),

                "length_difference":
                    length_difference,
            }
        )

    candidates.sort(
        key=lambda item: (
            item[
                "distance"
            ],
            -item[
                "frequency"
            ],
            item[
                "length_difference"
            ],
            item[
                "term"
            ],
        )
    )

    return candidates[
        :max(
            0,
            int(
                top_n
            ),
        )
    ]


# ============================================================
# 6. CORRECT ONE WORD
# ============================================================

def correct_word(
    word,
    vocabulary,
    max_distance=2,
    top_n=5,
):
    word = normalize_term(
        word
    )

    if not word:
        return {
            "original":
                "",

            "corrected":
                "",

            "changed":
                False,

            "known":
                False,

            "candidates":
                [],
        }

    frequency_map = (
        vocabulary
        if isinstance(
            vocabulary,
            dict,
        )
        else {
            normalize_term(
                term
            ): 0

            for term
            in vocabulary
        }
    )

    if word in frequency_map:
        return {
            "original":
                word,

            "corrected":
                word,

            "changed":
                False,

            "known":
                True,

            "candidates":
                [
                    {
                        "term":
                            word,

                        "distance":
                            0,

                        "frequency":
                            int(
                                frequency_map[
                                    word
                                ]
                            ),
                    }
                ],
        }

    candidates = find_candidates(
        word=
            word,

        vocabulary=
            frequency_map,

        max_distance=
            max_distance,

        top_n=
            top_n,
    )

    corrected = (
        candidates[0][
            "term"
        ]
        if candidates
        else word
    )

    return {
        "original":
            word,

        "corrected":
            corrected,

        "changed":
            (
                corrected
                != word
            ),

        "known":
            False,

        "candidates":
            candidates,
    }


# ============================================================
# 7. CORRECT QUERY
# ============================================================

def correct_query(
    query,
    index_data,
    max_distance=2,
    top_n=5,
):
    """
    Main E51 interface.

    Current correction scope:
        token-level spelling correction

    The output query is reconstructed from corrected IR tokens.
    """

    vocabulary = (
        vocabulary_from_index(
            index_data
        )
    )

    tokens = tokenize_words(
        query
        or ""
    )

    corrections = [
        correct_word(
            word=
                token,

            vocabulary=
                vocabulary,

            max_distance=
                max_distance,

            top_n=
                top_n,
        )

        for token
        in tokens
    ]

    corrected_tokens = [
        correction[
            "corrected"
        ]

        for correction
        in corrections
    ]

    changed = any(
        correction[
            "changed"
        ]

        for correction
        in corrections
    )

    return {
        "original_query":
            query
            or "",

        "corrected_query":
            " ".join(
                corrected_tokens
            ),

        "changed":
            changed,

        "tokens":
            tokens,

        "corrections":
            corrections,

        "vocabulary_size":
            len(
                vocabulary
            ),

        "max_distance":
            max_distance,
    }


# ============================================================
# 8. TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("E51 - EDIT DISTANCE / SPELLING TEST")
    print("=" * 70)

    source = "cancer"
    target = "caner"

    matrix = edit_distance_matrix(
        source,
        target,
    )

    print(
        f"Distance {source} -> {target}:",
        matrix[-1][-1],
    )

    demo_index = {
        "index": {
            "cancer": {
                "total_frequency": 100,
            },
            "cancers": {
                "total_frequency": 20,
            },
            "cancel": {
                "total_frequency": 10,
            },
            "patient": {
                "total_frequency": 80,
            },
        }
    }

    result = correct_query(
        query="caner patint",
        index_data=demo_index,
        max_distance=2,
    )

    print(
        "Original :",
        result[
            "original_query"
        ],
    )

    print(
        "Corrected:",
        result[
            "corrected_query"
        ],
    )

    print()

    for correction in result[
        "corrections"
    ]:
        print(
            correction
        )
