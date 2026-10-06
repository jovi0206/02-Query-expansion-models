from m02_text_processor import tokenize_words
from e5_01_spellcheck import (
    edit_distance as _e51_edit_distance,
    edit_distance_matrix as _e51_edit_distance_matrix,
    correct_word as _e51_correct_word,
    correct_query as _e51_correct_query,
)
from e5_02_semantic import (
    expand_word as _e52_expand_word,
    expand_query as _e52_expand_query,
)


# ============================================================
# MODULE 05 — QUERY PROCESSOR
# ============================================================
#
# Input:
#     User Query
#
# Process:
#     1. Validate query type
#     2. Normalize query terms
#     3. Detect Auto query type
#     4. Build a stable Query Plan
#
# Output:
#     query_plan
#
# M05 only answers:
#     "What does the user want to search for?"
#
# M05 does NOT:
#     Search documents
#     Read postings
#     Match phrases / sentences
#     Rank results
#     Display UI
#
# Next:
#     M06 Retrieval Engine consumes query_plan.
# ============================================================


VALID_QUERY_TYPES = {
    "auto",
    "word",
    "phrase",
    "sentence",
    "literal",
}


# ============================================================
# 1. BASIC NORMALIZATION
# ============================================================

def normalize_surface(token):
    """
    Normalize one query/token surface form.

    Example:
        Alzheimer's
        ALZHEIMER'S
        alzheimer’s

    become comparable lowercase surface forms.
    """

    if not token:
        return ""

    return (
        token
        .casefold()
        .replace("’", "'")
        .strip()
    )


# ============================================================
# 2. LIGHTWEIGHT LEXICAL BASE
# ============================================================
#
# Important:
#     This is retained from Project 1 only for lightweight
#     related matching.
#
#     It is NOT Porter stemming.
#     Project 2 Porter will be implemented as an Extension.
# ============================================================

def lexical_base(token):
    """
    Return a lightweight lexical base used by M06 related matching.

    Examples:
        Alzheimer's -> alzheimer
        activities  -> activity
        cancers     -> cancer

    This is intentionally NOT a full stemming algorithm.
    """

    token = normalize_surface(
        token
    )

    # Alzheimer's -> alzheimer
    if (
        token.endswith("'s")
        and len(token) > 2
    ):
        token = token[:-2]

    # activities -> activity
    if (
        len(token) > 4
        and token.endswith("ies")
    ):
        token = (
            token[:-3]
            + "y"
        )

    # boxes / matches / wishes
    elif (
        len(token) > 4
        and token.endswith(
            (
                "ches",
                "shes",
                "xes",
                "zes",
            )
        )
    ):
        token = token[:-2]

    # cancers -> cancer
    elif (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(
            (
                "ss",
                "us",
                "is",
            )
        )
    ):
        token = token[:-1]

    return token


# ============================================================
# 3. AUTO QUERY TYPE
# ============================================================

def detect_query_type(
    query,
    tokens=None,
):
    """
    Project 1 Auto mode rule:

        0 ~ 1 token
            -> word

        multiple tokens + sentence-like ending
        or >= 8 tokens
            -> sentence

        otherwise
            -> phrase
    """

    if tokens is None:
        tokens = tokenize_words(
            query
        )

    if len(tokens) <= 1:
        return "word"

    if (
        query.strip().endswith(
            (
                ".",
                "!",
                "?",
            )
        )
        or len(tokens) >= 8
    ):
        return "sentence"

    return "phrase"


# ============================================================
# 4. EMPTY QUERY PLAN
# ============================================================

def empty_query_plan(
    requested_query_type="auto",
):
    return {
        "original_query":
            "",

        "query":
            "",

        "requested_query_type":
            requested_query_type,

        "query_type":
            "empty",

        "auto_requested":
            (
                requested_query_type
                == "auto"
            ),

        "tokens":
            [],

        "normalized_terms":
            [],

        "lexical_terms":
            [],

        "is_empty":
            True,
    }


# ============================================================
# 5. BUILD QUERY PLAN
# ============================================================

def process_query(
    query,
    query_type="auto",
):
    """
    Convert a raw user query into a stable Query Plan.

    Input:
        query
        query_type

    Process:
        raw query
            ↓
        tokenize
            ↓
        normalize
            ↓
        resolve Auto / Word / Phrase / Sentence / Literal
            ↓
        Query Plan

    Output example:
        {
            "query": "physical activity",
            "query_type": "phrase",
            "tokens": ["physical", "activity"],
            "normalized_terms": ["physical", "activity"],
            "lexical_terms": ["physical", "activity"],
            ...
        }

    M06 should use this output instead of independently
    deciding what kind of query it received.
    """

    if query_type not in VALID_QUERY_TYPES:
        raise ValueError(
            "query_type must be "
            "auto, word, phrase, sentence, or literal"
        )

    original_query = (
        query
        if query is not None
        else ""
    )

    query = original_query.strip()

    if not query:
        return empty_query_plan(
            requested_query_type=
                query_type
        )

    tokens = tokenize_words(
        query
    )

    auto_requested = (
        query_type
        == "auto"
    )

    if auto_requested:
        resolved_query_type = (
            detect_query_type(
                query,
                tokens=tokens,
            )
        )
    else:
        resolved_query_type = (
            query_type
        )

    normalized_terms = [
        normalize_surface(
            token
        )

        for token
        in tokens
    ]

    lexical_terms = [
        lexical_base(
            token
        )

        for token
        in tokens
    ]

    query_plan = {
        "original_query":
            original_query,

        # Effective query used by M06.
        # Later E5x query-correction extensions can produce
        # another query and call process_query() again.
        "query":
            query,

        "requested_query_type":
            query_type,

        "query_type":
            resolved_query_type,

        "auto_requested":
            auto_requested,

        "tokens":
            tokens,

        "normalized_terms":
            normalized_terms,

        "lexical_terms":
            lexical_terms,

        "is_empty":
            False,
    }

    return query_plan


# ============================================================
# E51 / E52 INTEGRATION — QUERY EXPANSION SUPPORT
# ============================================================
#
# M05 remains responsible for query processing.
# E51 supplies Edit Distance / spelling correction.
# E52 supplies Word2Vec-based semantic expansion.
#
# These functions do not automatically alter baseline search.
# M08 / future M09 can explicitly choose whether to use an
# original, corrected, or expanded query.
# ============================================================

def calculate_edit_distance(source, target, max_distance=None):
    """Calculate Levenshtein distance through E51."""
    return _e51_edit_distance(
        source=source,
        target=target,
        max_distance=max_distance,
    )


def build_edit_distance_matrix(source, target):
    """Return the full E51 DP matrix for teaching / demo use."""
    return _e51_edit_distance_matrix(
        source=source,
        target=target,
    )


def correct_query_word(
    word,
    vocabulary,
    max_distance=2,
    top_n=5,
):
    """Correct one word using an E51 vocabulary."""
    return _e51_correct_word(
        word=word,
        vocabulary=vocabulary,
        max_distance=max_distance,
        top_n=top_n,
    )


def correct_spelling(
    query,
    index_data,
    max_distance=2,
    top_n=5,
):
    """Correct a complete query against the M04 index vocabulary."""
    return _e51_correct_query(
        query=query,
        index_data=index_data,
        max_distance=max_distance,
        top_n=top_n,
    )


def expand_semantic_word(
    model,
    word,
    top_n=5,
    min_similarity=0.5,
):
    """Expand one word through E52 + E31 Word2Vec."""
    return _e52_expand_word(
        model=model,
        word=word,
        top_n=top_n,
        min_similarity=min_similarity,
    )


def expand_semantic_query(
    query,
    model,
    top_n_per_word=3,
    min_similarity=0.5,
):
    """Expand a complete query through E52 + E31 Word2Vec."""
    return _e52_expand_query(
        query=query,
        model=model,
        top_n_per_word=top_n_per_word,
        min_similarity=min_similarity,
    )


# ============================================================
# 6. TEST
# ============================================================

if __name__ == "__main__":

    test_queries = [
        ("cancer", "auto"),
        ("physical activity", "auto"),
        (
            "This study systematically reviews the evidence.",
            "auto",
        ),
        ("eurol Sc", "literal"),
    ]

    print("=" * 80)
    print("MODULE 05 - QUERY PROCESSOR TEST")
    print("=" * 80)

    for query, query_type in test_queries:

        plan = process_query(
            query=query,
            query_type=query_type,
        )

        print()
        print(f"Original Query : {plan['original_query']}")
        print(f"Requested Type : {plan['requested_query_type']}")
        print(f"Resolved Type  : {plan['query_type']}")
        print(f"Tokens         : {plan['tokens']}")
        print(f"Normalized     : {plan['normalized_terms']}")
        print(f"Lexical Base   : {plan['lexical_terms']}")
        print("-" * 80)
