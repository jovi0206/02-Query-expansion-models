from m02_text_processor import (
    tokenize_words,
)

from e3_01_word2vec import (
    normalize_word,
)


# ============================================================
# EXTENSION E52 — SEMANTIC QUERY EXPANSION
# ============================================================
#
# Parent Module:
#     M05 — Query Processing
#
# Dependency:
#     E31 — Word2Vec
#
# Input:
#     Query
#     Trained Word2Vec model
#
# Process:
#     Find semantically similar words
#
# Output:
#     Original terms
#     Expansion terms
#     Expanded query
#
# Important:
#     This extension is optional for the assignment baseline.
#     Word2Vec itself is required by Project 2.
#
#     E52 turns Word2Vec representation into an actual
#     query-expansion function.
# ============================================================


# ============================================================
# 1. EXPAND ONE WORD
# ============================================================

def expand_word(
    model,
    word,
    top_n=5,
    min_similarity=0.5,
):
    word = normalize_word(
        word
    )

    if not word:
        return {
            "word":
                "",

            "known":
                False,

            "expansions":
                [],
        }

    if word not in model.wv:
        return {
            "word":
                word,

            "known":
                False,

            "expansions":
                [],
        }

    raw_candidates = (
        model.wv.most_similar(
            word,
            topn=
                max(
                    top_n
                    * 3,
                    top_n,
                ),
        )
    )

    expansions = []

    seen = {
        word
    }

    for (
        candidate,
        similarity,
    ) in raw_candidates:

        candidate = normalize_word(
            candidate
        )

        similarity = float(
            similarity
        )

        if not candidate:
            continue

        if candidate in seen:
            continue

        if (
            similarity
            < min_similarity
        ):
            continue

        seen.add(
            candidate
        )

        expansions.append(
            {
                "word":
                    candidate,

                "similarity":
                    round(
                        similarity,
                        6,
                    ),
            }
        )

        if (
            len(
                expansions
            )
            >= top_n
        ):
            break

    return {
        "word":
            word,

        "known":
            True,

        "expansions":
            expansions,
    }


# ============================================================
# 2. EXPAND QUERY
# ============================================================

def expand_query(
    query,
    model,
    top_n_per_word=3,
    min_similarity=0.5,
):
    """
    Main E52 interface.

    Example:
        query:
            obesity treatment

        possible semantic expansion:
            obesity treatment overweight therapy ...
    """

    original_tokens = [
        normalize_word(
            token
        )

        for token
        in tokenize_words(
            query
            or ""
        )
    ]

    original_tokens = [
        token

        for token
        in original_tokens

        if token
    ]

    per_word = []

    expansion_terms = []

    seen = set(
        original_tokens
    )

    for token in original_tokens:

        result = expand_word(
            model=
                model,

            word=
                token,

            top_n=
                top_n_per_word,

            min_similarity=
                min_similarity,
        )

        per_word.append(
            result
        )

        for expansion in result[
            "expansions"
        ]:

            candidate = expansion[
                "word"
            ]

            if candidate in seen:
                continue

            seen.add(
                candidate
            )

            expansion_terms.append(
                {
                    "source_word":
                        token,

                    "word":
                        candidate,

                    "similarity":
                        expansion[
                            "similarity"
                        ],
                }
            )

    expanded_tokens = (
        original_tokens
        + [
            item[
                "word"
            ]

            for item
            in expansion_terms
        ]
    )

    return {
        "original_query":
            query
            or "",

        "original_tokens":
            original_tokens,

        "expansion_terms":
            expansion_terms,

        "expanded_tokens":
            expanded_tokens,

        "expanded_query":
            " ".join(
                expanded_tokens
            ),

        "changed":
            bool(
                expansion_terms
            ),

        "top_n_per_word":
            top_n_per_word,

        "min_similarity":
            min_similarity,

        "per_word":
            per_word,
    }


# ============================================================
# 3. SIMPLE FAKE MODEL TEST
# ============================================================

class _DemoWV:

    def __init__(
        self,
    ):
        self.words = {
            "obesity",
            "treatment",
        }

    def __contains__(
        self,
        word,
    ):
        return (
            word
            in self.words
        )

    def most_similar(
        self,
        word,
        topn=10,
    ):
        if word == "obesity":
            return [
                ("overweight", 0.88),
                ("adiposity", 0.82),
                ("diabetes", 0.58),
            ][
                :topn
            ]

        if word == "treatment":
            return [
                ("therapy", 0.90),
                ("intervention", 0.76),
                ("management", 0.70),
            ][
                :topn
            ]

        return []


class _DemoModel:

    def __init__(
        self,
    ):
        self.wv = _DemoWV()


if __name__ == "__main__":

    demo_model = (
        _DemoModel()
    )

    result = expand_query(
        query=
            "obesity treatment",

        model=
            demo_model,

        top_n_per_word=
            2,

        min_similarity=
            0.6,
    )

    print("=" * 70)
    print("E52 - SEMANTIC QUERY EXPANSION TEST")
    print("=" * 70)

    print(
        "Original:",
        result[
            "original_query"
        ],
    )

    print(
        "Expanded:",
        result[
            "expanded_query"
        ],
    )

    print()

    for item in result[
        "expansion_terms"
    ]:
        print(
            item
        )
