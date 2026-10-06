import re
from collections import Counter

from e2_01_porter import porter_stem


# ============================================================
# EXTENSION E22 — PREPROCESSING EXPERIMENT
# ============================================================
#
# Parent Module:
#     M02 — Text Processing
#
# Purpose:
#     Build the four preprocessing conditions required by Project 2:
#
#     A — Basic preprocessing
#         Tokenization + case folding
#
#     B — Remove punctuation
#         A + punctuation-aware tokenization
#
#     C — Remove stopwords
#         B + stopword removal
#
#     D — Stemming
#         C + Porter stemming (E21)
#
# Input:
#     M02 processed_documents
#
# Output:
#     Corpus-level statistics for every condition:
#         document count
#         total tokens
#         unique terms
#         average tokens/document
#         collection frequency (CF)
#         document frequency (DF)
#         Top-N terms
#
# Design note:
#     E22 does not import M02.  M02 passes its tokenizer callback into E22,
#     which avoids a circular import and keeps tokenization policy owned by M02.
# ============================================================


# A fixed, self-contained English stopword set.
# The exact list is intentionally stored in source so experiments are
# reproducible without downloading an external NLP corpus.
DEFAULT_ENGLISH_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
})


CONDITION_ORDER = (
    "A_basic",
    "B_no_punctuation",
    "C_no_stopwords",
    "D_porter_stemming",
)

# Page 08 Porter demonstration compares the same E21 Porter stemmer
# against three different preprocessing token streams.  These are
# additional comparisons only; they do NOT add new formal A-D conditions.
PORTER_DEMO_ORDER = (
    "A_basic",
    "B_no_punctuation",
    "C_no_stopwords",
)


CONDITION_METADATA = {
    "A_basic": {
        "label": "Condition A — Basic preprocessing",
        "steps": [
            "Whitespace tokenization",
            "Case folding",
        ],
        "description": (
            "Lowercase whitespace-delimited tokens. Punctuation is intentionally "
            "left attached so Condition B has a measurable punctuation-removal effect."
        ),
    },
    "B_no_punctuation": {
        "label": "Condition B — Remove punctuation",
        "steps": [
            "M02 punctuation-aware tokenization",
            "Case folding",
        ],
        "description": (
            "Uses the parent M02 tokenizer, which removes punctuation delimiters "
            "while preserving supported scientific intra-token forms."
        ),
    },
    "C_no_stopwords": {
        "label": "Condition C — Remove stopwords",
        "steps": [
            "Condition B",
            "English stopword removal",
        ],
        "description": "Condition B tokens with the fixed E22 English stopword list removed.",
    },
    "D_porter_stemming": {
        "label": "Condition D — Porter stemming",
        "steps": [
            "Condition C",
            "Porter stemming (E21)",
        ],
        "description": "Condition C tokens transformed with the Project 2 E21 Porter stemmer.",
    },
}


def get_document_id(document):
    return (
        document.get("pmcid")
        or document.get("pmid")
        or document.get("filename", "")
        or "unknown"
    )


def get_document_texts(processed_document):
    """Return M02 search-segment text in document order."""
    texts = []

    for segment in processed_document.get("search_segments", []):
        text = str(segment.get("text", "") or "").strip()
        if text:
            texts.append(text)

    # Compatibility fallback for an older processed document.
    if not texts:
        text = str(processed_document.get("search_text", "") or "").strip()
        if text:
            texts.append(text)

    return texts


def tokenize_basic(text):
    """Condition A tokenizer: whitespace boundaries only, then lowercase."""
    return [
        token.casefold()
        for token in re.findall(r"\S+", str(text or ""))
        if token.strip()
    ]


def tokenize_without_punctuation(text, punctuation_tokenizer):
    """Condition B tokenizer using M02's punctuation-aware tokenizer callback."""
    if punctuation_tokenizer is None:
        raise ValueError(
            "punctuation_tokenizer is required. M02 should pass tokenize_words()."
        )

    return [
        str(token).casefold().strip()
        for token in punctuation_tokenizer(str(text or ""))
        if str(token).strip()
    ]


def remove_stopwords(tokens, stopwords=None):
    stopwords = DEFAULT_ENGLISH_STOPWORDS if stopwords is None else {
        str(term).casefold().strip()
        for term in stopwords
        if str(term).strip()
    }

    return [
        token
        for token in tokens
        if token.casefold() not in stopwords
    ]


def stem_tokens(tokens):
    stems = []

    for token in tokens:
        stem = porter_stem(token)
        if stem:
            stems.append(stem)

    return stems


def build_condition_summary(
    condition_key,
    document_token_lists,
    top_n=50,
    metadata=None,
):
    collection_frequency = Counter()
    document_frequency = Counter()
    document_token_counts = []

    for document_id, tokens in document_token_lists:
        collection_frequency.update(tokens)
        document_frequency.update(set(tokens))
        document_token_counts.append(
            {
                "document_id": document_id,
                "token_count": len(tokens),
            }
        )

    document_count = len(document_token_lists)
    total_tokens = sum(collection_frequency.values())
    unique_terms = len(collection_frequency)

    sorted_terms = sorted(
        collection_frequency.items(),
        key=lambda item: (-item[1], item[0]),
    )

    if top_n is not None:
        sorted_terms = sorted_terms[:max(0, int(top_n))]

    top_terms = [
        {
            "rank": rank,
            "term": term,
            "collection_frequency": cf,
            "document_frequency": int(document_frequency.get(term, 0)),
        }
        for rank, (term, cf) in enumerate(sorted_terms, start=1)
    ]

    metadata = metadata or CONDITION_METADATA.get(condition_key) or {
        "label": condition_key,
        "steps": [],
        "description": "",
    }

    return {
        "condition": condition_key,
        "label": metadata["label"],
        "steps": list(metadata["steps"]),
        "description": metadata["description"],
        "document_count": document_count,
        "total_tokens": total_tokens,
        "unique_terms": unique_terms,
        "average_tokens_per_document": (
            total_tokens / document_count
            if document_count
            else 0.0
        ),
        "collection_frequency": dict(collection_frequency),
        "document_frequency": dict(document_frequency),
        "top_terms": top_terms,
        "document_token_counts": document_token_counts,
    }


def run_preprocessing_experiment(
    processed_documents,
    punctuation_tokenizer,
    top_n=50,
    stopwords=None,
):
    """
    Run the four formal Project 2 preprocessing conditions on the same corpus
    and prepare the additional Page 08 Porter before/after comparisons.

    Formal A-D experiment (unchanged):
        A = basic
        B = punctuation-aware tokenization
        C = B + stopword removal
        D = C + Porter stemming

    Additional Porter demo (does not redefine A-D):
        A -> A + Porter
        B -> B + Porter
        C -> C + Porter (= formal D)

    The same E21 Porter implementation is used in every comparison.  This
    makes the effect of applying Porter to different token streams directly
    visible without changing the required A-D experiment.
    """
    processed_documents = list(processed_documents or [])

    effective_stopwords = (
        DEFAULT_ENGLISH_STOPWORDS
        if stopwords is None
        else frozenset(
            str(term).casefold().strip()
            for term in stopwords
            if str(term).strip()
        )
    )

    condition_documents = {
        key: []
        for key in CONDITION_ORDER
    }

    # Only A+Porter and B+Porter need separate token streams.
    # C+Porter is exactly the formal D stream, so we intentionally reuse D
    # rather than calculating/storing a second copy.
    porter_demo_documents = {
        "A_basic": [],
        "B_no_punctuation": [],
    }

    for document in processed_documents:
        document_id = get_document_id(document)
        texts = get_document_texts(document)

        basic_tokens = []
        punctuation_tokens = []

        for text in texts:
            basic_tokens.extend(tokenize_basic(text))
            punctuation_tokens.extend(
                tokenize_without_punctuation(
                    text,
                    punctuation_tokenizer=punctuation_tokenizer,
                )
            )

        no_stopwords_tokens = remove_stopwords(
            punctuation_tokens,
            stopwords=effective_stopwords,
        )
        stemmed_tokens = stem_tokens(no_stopwords_tokens)

        condition_documents["A_basic"].append((document_id, basic_tokens))
        condition_documents["B_no_punctuation"].append((document_id, punctuation_tokens))
        condition_documents["C_no_stopwords"].append((document_id, no_stopwords_tokens))
        condition_documents["D_porter_stemming"].append((document_id, stemmed_tokens))

        porter_demo_documents["A_basic"].append(
            (document_id, stem_tokens(basic_tokens))
        )
        porter_demo_documents["B_no_punctuation"].append(
            (document_id, stem_tokens(punctuation_tokens))
        )

    conditions = {
        key: build_condition_summary(
            key,
            condition_documents[key],
            top_n=top_n,
        )
        for key in CONDITION_ORDER
    }

    basic_vocab = conditions["A_basic"]["unique_terms"]
    basic_tokens = conditions["A_basic"]["total_tokens"]

    comparison = []
    previous_tokens = None
    previous_vocab = None

    for key in CONDITION_ORDER:
        result = conditions[key]
        total_tokens = result["total_tokens"]
        unique_terms = result["unique_terms"]

        comparison.append(
            {
                "condition": key,
                "label": result["label"],
                "documents": result["document_count"],
                "total_tokens": total_tokens,
                "unique_terms": unique_terms,
                "average_tokens_per_document": result[
                    "average_tokens_per_document"
                ],
                "token_change_from_A": total_tokens - basic_tokens,
                "token_change_ratio_from_A": (
                    (total_tokens - basic_tokens) / basic_tokens
                    if basic_tokens
                    else 0.0
                ),
                "vocabulary_change_from_A": unique_terms - basic_vocab,
                "vocabulary_change_ratio_from_A": (
                    (unique_terms - basic_vocab) / basic_vocab
                    if basic_vocab
                    else 0.0
                ),
                "token_change_from_previous": (
                    None
                    if previous_tokens is None
                    else total_tokens - previous_tokens
                ),
                "vocabulary_change_from_previous": (
                    None
                    if previous_vocab is None
                    else unique_terms - previous_vocab
                ),
            }
        )

        previous_tokens = total_tokens
        previous_vocab = unique_terms

    # --------------------------------------------------------
    # Additional Porter before/after comparisons for Page 08.
    # --------------------------------------------------------
    porter_demo = {
        "comparison_order": list(PORTER_DEMO_ORDER),
        "algorithm": "E21 Porter stemming",
        "formal_conditions_unchanged": True,
        "comparisons": {},
    }

    for key in PORTER_DEMO_ORDER:
        before = conditions[key]

        if key == "C_no_stopwords":
            # Formal Condition D is exactly Condition C + Porter.
            after_condition_key = "D_porter_stemming"
            after = conditions[after_condition_key]
            after_result = None
            formal_equivalent = after_condition_key
        else:
            after_condition_key = None
            formal_equivalent = None
            after = build_condition_summary(
                f"{key}_porter_demo",
                porter_demo_documents[key],
                top_n=top_n,
                metadata={
                    "label": f"{before['label']} + Porter",
                    "steps": list(before.get("steps", [])) + ["Porter stemming (E21)"],
                    "description": (
                        "Additional Porter demonstration only; the formal A-D "
                        "preprocessing experiment is unchanged."
                    ),
                },
            )
            after_result = after

        before_vocab = int(before.get("unique_terms", 0) or 0)
        after_vocab = int(after.get("unique_terms", 0) or 0)
        vocab_reduction = before_vocab - after_vocab

        porter_demo["comparisons"][key] = {
            "before_condition_key": key,
            "after_condition_key": after_condition_key,
            "formal_equivalent": formal_equivalent,
            "after_result": after_result,
            "document_count": int(before.get("document_count", 0) or 0),
            "token_count_before": int(before.get("total_tokens", 0) or 0),
            "token_count_after": int(after.get("total_tokens", 0) or 0),
            "vocabulary_before": before_vocab,
            "vocabulary_after": after_vocab,
            "vocabulary_reduction": vocab_reduction,
            "vocabulary_reduction_ratio": (
                vocab_reduction / before_vocab
                if before_vocab
                else 0.0
            ),
        }

    return {
        "experiment": "project2_preprocessing",
        "document_count": len(processed_documents),
        "condition_order": list(CONDITION_ORDER),
        "stopword_count": len(effective_stopwords),
        "stopword_method": "Fixed embedded English stopword list",
        "conditions": conditions,
        "comparison": comparison,
        "porter_demo": porter_demo,
    }


if __name__ == "__main__":
    demo_documents = [
        {
            "pmid": "1",
            "search_segments": [
                {"text": "The DATA, data. models are modeling studies."},
            ],
        },
        {
            "pmid": "2",
            "search_segments": [
                {"text": "Data-driven treatment and treatments improve outcomes."},
            ],
        },
    ]

    demo_tokenizer = re.compile(
        r"\d+(?:[.,]\d+)*(?:%|[A-Za-z]+)?|[^\W_]+(?:[.'’/‐‑‒–—-][^\W_]+)*%?",
        re.UNICODE,
    ).findall

    result = run_preprocessing_experiment(
        demo_documents,
        punctuation_tokenizer=demo_tokenizer,
    )

    print("E22 - PREPROCESSING EXPERIMENT TEST")
    for row in result["comparison"]:
        print(row)
