import re
from collections import Counter


# ============================================================
# EXTENSION E21 — PORTER STEMMER
# ============================================================
#
# Parent Module:
#     M02 — Content Preprocessing
#
# Input:
#     M02 processed_document / processed_documents
#
# Uses:
#     search_segments[*]["words"]
#
# Output:
#     Original tokens
#     Stemmed tokens
#     Original frequency
#     Stemmed frequency
#     Vocabulary comparison
#
# Purpose:
#     Project 2 — Query Expansion Models
#
#     Implement Porter's stemming algorithm as a functional
#     module and prepare data for later Zipf comparison.
#
# Important:
#     This is the classic Porter-style stemming process.
#     It is NOT M05's lightweight lexical_base().
#
#     Porter stemming is applied only to English alphabetic
#     components. Numbers / symbols / non-English components
#     are preserved in normalized form.
# ============================================================


# ============================================================
# 1. NORMALIZATION
# ============================================================

SEPARATOR_PATTERN = re.compile(
    r"([/'’‐‑‒–—-]+)"
)

ASCII_WORD_PATTERN = re.compile(
    r"^[a-z]+$"
)


def normalize_token(token):
    if not token:
        return ""

    return (
        str(token)
        .casefold()
        .replace("’", "'")
        .strip()
    )


# ============================================================
# 2. PORTER BASIC DEFINITIONS
# ============================================================

VOWELS = {
    "a",
    "e",
    "i",
    "o",
    "u",
}


def is_consonant(
    word,
    index,
):
    """
    Porter definition of consonant.

    y is:
        consonant at position 0
        opposite of the previous character afterwards
    """

    char = word[
        index
    ]

    if char in VOWELS:
        return False

    if char != "y":
        return True

    if index == 0:
        return True

    return not is_consonant(
        word,
        index - 1,
    )


def measure(
    word,
):
    """
    Porter measure m:
        [C](VC)^m[V]
    """

    if not word:
        return 0

    m = 0
    index = 0
    length = len(
        word
    )

    # Skip initial consonants.
    while (
        index < length
        and is_consonant(
            word,
            index,
        )
    ):
        index += 1

    while index < length:

        # Skip vowels.
        while (
            index < length
            and not is_consonant(
                word,
                index,
            )
        ):
            index += 1

        if index >= length:
            break

        # One VC sequence completed.
        m += 1

        # Skip consonants.
        while (
            index < length
            and is_consonant(
                word,
                index,
            )
        ):
            index += 1

    return m


def contains_vowel(
    word,
):
    return any(
        not is_consonant(
            word,
            index,
        )

        for index
        in range(
            len(word)
        )
    )


def ends_with_double_consonant(
    word,
):
    if len(word) < 2:
        return False

    if (
        word[-1]
        != word[-2]
    ):
        return False

    return is_consonant(
        word,
        len(word) - 1,
    )


def ends_cvc(
    word,
):
    """
    Porter *o condition:
        consonant-vowel-consonant
        final consonant is not w, x, y
    """

    if len(word) < 3:
        return False

    last = len(
        word
    ) - 1

    return (
        is_consonant(
            word,
            last - 2,
        )
        and not is_consonant(
            word,
            last - 1,
        )
        and is_consonant(
            word,
            last,
        )
        and word[
            last
        ]
        not in {
            "w",
            "x",
            "y",
        }
    )


# ============================================================
# 3. PORTER STEP 1
# ============================================================

def step_1a(
    word,
):
    if word.endswith(
        "sses"
    ):
        return word[:-2]

    if word.endswith(
        "ies"
    ):
        return (
            word[:-3]
            + "i"
        )

    if word.endswith(
        "ss"
    ):
        return word

    if word.endswith(
        "s"
    ):
        return word[:-1]

    return word


def step_1b(
    word,
):
    if word.endswith(
        "eed"
    ):
        stem = word[:-3]

        if measure(
            stem
        ) > 0:
            return (
                stem
                + "ee"
            )

        return word

    removed = False
    stem = word

    if word.endswith(
        "ed"
    ):
        candidate = word[:-2]

        if contains_vowel(
            candidate
        ):
            stem = candidate
            removed = True

    elif word.endswith(
        "ing"
    ):
        candidate = word[:-3]

        if contains_vowel(
            candidate
        ):
            stem = candidate
            removed = True

    if not removed:
        return word

    if stem.endswith(
        (
            "at",
            "bl",
            "iz",
        )
    ):
        return (
            stem
            + "e"
        )

    if (
        ends_with_double_consonant(
            stem
        )
        and stem[-1]
        not in {
            "l",
            "s",
            "z",
        }
    ):
        return stem[:-1]

    if (
        measure(
            stem
        )
        == 1
        and ends_cvc(
            stem
        )
    ):
        return (
            stem
            + "e"
        )

    return stem


def step_1c(
    word,
):
    if (
        word.endswith(
            "y"
        )
        and len(word) > 1
    ):
        stem = word[:-1]

        if contains_vowel(
            stem
        ):
            return (
                stem
                + "i"
            )

    return word


# ============================================================
# 4. PORTER STEP 2
# ============================================================

STEP_2_RULES = [
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
    ("logi", "log"),
]


def step_2(
    word,
):
    for (
        suffix,
        replacement,
    ) in STEP_2_RULES:

        if not word.endswith(
            suffix
        ):
            continue

        stem = word[
            :-len(
                suffix
            )
        ]

        if measure(
            stem
        ) > 0:
            return (
                stem
                + replacement
            )

        return word

    return word


# ============================================================
# 5. PORTER STEP 3
# ============================================================

STEP_3_RULES = [
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
]


def step_3(
    word,
):
    for (
        suffix,
        replacement,
    ) in STEP_3_RULES:

        if not word.endswith(
            suffix
        ):
            continue

        stem = word[
            :-len(
                suffix
            )
        ]

        if measure(
            stem
        ) > 0:
            return (
                stem
                + replacement
            )

        return word

    return word


# ============================================================
# 6. PORTER STEP 4
# ============================================================

STEP_4_SUFFIXES = [
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
]


def step_4(
    word,
):
    # Special Porter ion rule.
    if word.endswith(
        "ion"
    ):
        stem = word[:-3]

        if (
            measure(
                stem
            ) > 1
            and stem.endswith(
                (
                    "s",
                    "t",
                )
            )
        ):
            return stem

        return word

    for suffix in STEP_4_SUFFIXES:

        if not word.endswith(
            suffix
        ):
            continue

        stem = word[
            :-len(
                suffix
            )
        ]

        if measure(
            stem
        ) > 1:
            return stem

        return word

    return word


# ============================================================
# 7. PORTER STEP 5
# ============================================================

def step_5a(
    word,
):
    if not word.endswith(
        "e"
    ):
        return word

    stem = word[:-1]
    stem_measure = measure(
        stem
    )

    if stem_measure > 1:
        return stem

    if (
        stem_measure
        == 1
        and not ends_cvc(
            stem
        )
    ):
        return stem

    return word


def step_5b(
    word,
):
    if (
        word.endswith(
            "ll"
        )
        and measure(
            word
        ) > 1
    ):
        return word[:-1]

    return word


# ============================================================
# 8. STEM ONE ENGLISH WORD
# ============================================================

def porter_stem_word(
    word,
):
    """
    Apply the Porter algorithm to one lowercase ASCII word.
    """

    word = normalize_token(
        word
    )

    if (
        len(word) <= 2
        or not ASCII_WORD_PATTERN.fullmatch(
            word
        )
    ):
        return word

    word = step_1a(
        word
    )

    word = step_1b(
        word
    )

    word = step_1c(
        word
    )

    word = step_2(
        word
    )

    word = step_3(
        word
    )

    word = step_4(
        word
    )

    word = step_5a(
        word
    )

    word = step_5b(
        word
    )

    return word


# ============================================================
# 9. STEM ONE M02 TOKEN
# ============================================================

def porter_stem(
    token,
):
    """
    Stem one M02 search token.

    Compound tokens are handled component-by-component while
    preserving separators.

    Examples:
        cancer-related
        PI3K/Akt
        Alzheimer's
    """

    token = normalize_token(
        token
    )

    if not token:
        return ""

    parts = SEPARATOR_PATTERN.split(
        token
    )

    stemmed_parts = []

    for part in parts:

        if not part:
            continue

        if SEPARATOR_PATTERN.fullmatch(
            part
        ):
            stemmed_parts.append(
                part.replace(
                    "’",
                    "'",
                )
            )
            continue

        if ASCII_WORD_PATTERN.fullmatch(
            part
        ):
            stemmed_parts.append(
                porter_stem_word(
                    part
                )
            )
        else:
            stemmed_parts.append(
                part
            )

    return "".join(
        stemmed_parts
    )


# ============================================================
# 10. M02 TOKEN EXTRACTION
# ============================================================

def get_document_tokens(
    processed_document,
):
    """
    Use M02 Search Corpus tokens, not Document Statistics words.

    This preserves the same token definition used by the IR
    pipeline and avoids mixing whitespace statistics tokens
    with search/index tokens.
    """

    tokens = []

    for segment in processed_document.get(
        "search_segments",
        [],
    ):
        tokens.extend(
            segment.get(
                "words",
                [],
            )
        )

    return tokens


def stem_tokens(
    tokens,
):
    return [
        porter_stem(
            token
        )

        for token
        in tokens
    ]


# ============================================================
# 11. ONE DOCUMENT
# ============================================================

def stem_document(
    processed_document,
):
    original_tokens = (
        get_document_tokens(
            processed_document
        )
    )

    normalized_tokens = [
        normalize_token(
            token
        )

        for token
        in original_tokens
    ]

    stemmed_tokens = (
        stem_tokens(
            original_tokens
        )
    )

    original_frequency = Counter(
        normalized_tokens
    )

    stemmed_frequency = Counter(
        stemmed_tokens
    )

    document_id = (
        processed_document.get(
            "pmcid"
        )
        or processed_document.get(
            "pmid"
        )
        or processed_document.get(
            "filename",
            ""
        )
    )

    return {
        "document_id":
            document_id,

        "title":
            processed_document.get(
                "title",
                "",
            ),

        "original_tokens":
            normalized_tokens,

        "stemmed_tokens":
            stemmed_tokens,

        "original_frequency":
            dict(
                original_frequency
            ),

        "stemmed_frequency":
            dict(
                stemmed_frequency
            ),

        "token_count":
            len(
                normalized_tokens
            ),

        "original_vocab_size":
            len(
                original_frequency
            ),

        "stemmed_vocab_size":
            len(
                stemmed_frequency
            ),
    }


# ============================================================
# 12. CORPUS
# ============================================================

def stem_corpus(
    processed_documents,
):
    """
    Main E21 corpus interface.

    Output can later be consumed by E41 Zipf analysis.
    """

    document_results = [
        stem_document(
            document
        )

        for document
        in (
            processed_documents
            or []
        )
    ]

    original_frequency = Counter()
    stemmed_frequency = Counter()

    for result in document_results:
        original_frequency.update(
            result[
                "original_frequency"
            ]
        )

        stemmed_frequency.update(
            result[
                "stemmed_frequency"
            ]
        )

    token_count = sum(
        original_frequency.values()
    )

    original_vocab_size = len(
        original_frequency
    )

    stemmed_vocab_size = len(
        stemmed_frequency
    )

    vocabulary_reduction = (
        original_vocab_size
        - stemmed_vocab_size
    )

    reduction_ratio = (
        vocabulary_reduction
        / original_vocab_size
        if original_vocab_size
        else 0.0
    )

    return {
        "documents":
            document_results,

        "document_count":
            len(
                document_results
            ),

        "token_count":
            token_count,

        "original_frequency":
            dict(
                original_frequency
            ),

        "stemmed_frequency":
            dict(
                stemmed_frequency
            ),

        "original_vocab_size":
            original_vocab_size,

        "stemmed_vocab_size":
            stemmed_vocab_size,

        "vocabulary_reduction":
            vocabulary_reduction,

        "vocabulary_reduction_ratio":
            round(
                reduction_ratio,
                6,
            ),
    }


# ============================================================
# 13. TEST
# ============================================================

if __name__ == "__main__":

    examples = [
        "caresses",
        "ponies",
        "ties",
        "caress",
        "cats",
        "feed",
        "agreed",
        "plastered",
        "motoring",
        "sing",
        "relational",
        "conditional",
        "rational",
        "valenci",
        "hesitanci",
        "digitizer",
        "vietnamization",
        "triplicate",
        "formalize",
        "electricical",
        "hopeful",
        "goodness",
    ]

    print(
        "=" * 70
    )

    print(
        "E21 - PORTER STEMMER TEST"
    )

    print(
        "=" * 70
    )

    for word in examples:
        print(
            f"{word:20s} -> "
            f"{porter_stem(word)}"
        )
