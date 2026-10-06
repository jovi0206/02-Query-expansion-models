from pathlib import Path


# ============================================================
# EXTENSION E31 — WORD2VEC
# ============================================================
#
# Parent Module:
#     M03 — Representation
#
# Input:
#     M03 positioned_documents
#
# Uses:
#     segments
#     sentence boundaries
#     word positions
#
# Output:
#     Word2Vec model
#     Training metadata
#     Similar-word lookup
#
# Project 2:
#     Word Embedding Technique (Word2Vec)
#
# Models:
#     CBOW
#     Skip-gram
#
# Default:
#     Skip-gram
#
# Why default Skip-gram:
#     Biomedical corpora often contain meaningful lower-frequency
#     terms. Skip-gram is a reasonable default for that setting.
#
# Important:
#     Both CBOW and Skip-gram remain selectable so the project
#     can change the model later without redesigning this module.
#
# Dependency:
#     gensim
#
#     pip install gensim
# ============================================================


# ============================================================
# 1. TOKEN NORMALIZATION
# ============================================================

def normalize_word(
    word,
):
    if not word:
        return ""

    return (
        str(word)
        .casefold()
        .replace(
            "’",
            "'",
        )
        .strip()
    )


# ============================================================
# 2. SENTENCE TOKEN EXTRACTION FROM M03
# ============================================================

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
        in segment.get(
            "words",
            [],
        )

        if (
            start
            <= word[
                "global_char_start"
            ]
            < end
        )
    ]


def prepare_document_sentences(
    positioned_document,
    include_unsentenced=False,
    lowercase=True,
    min_tokens=2,
):
    """
    Convert one M03 positioned document into ordered token
    sequences suitable for Word2Vec training.

    Sentence boundaries are preserved.

    By default, segments without sentence boundaries are skipped.
    This avoids training on structural labels as if they were
    normal prose.

    Set include_unsentenced=True only if needed later.
    """

    prepared = []

    for segment in positioned_document.get(
        "segments",
        [],
    ):

        sentences = segment.get(
            "sentences",
            [],
        )

        if sentences:

            for sentence in sentences:

                words = get_sentence_words(
                    segment,
                    sentence,
                )

                tokens = [
                    (
                        normalize_word(
                            word["text"]
                        )
                        if lowercase
                        else word["text"]
                    )

                    for word
                    in words

                    if word.get(
                        "text"
                    )
                ]

                tokens = [
                    token

                    for token
                    in tokens

                    if token
                ]

                if (
                    len(tokens)
                    >= min_tokens
                ):
                    prepared.append(
                        tokens
                    )

            continue

        if not include_unsentenced:
            continue

        tokens = [
            (
                normalize_word(
                    word["text"]
                )
                if lowercase
                else word["text"]
            )

            for word
            in segment.get(
                "words",
                [],
            )

            if word.get(
                "text"
            )
        ]

        tokens = [
            token

            for token
            in tokens

            if token
        ]

        if (
            len(tokens)
            >= min_tokens
        ):
            prepared.append(
                tokens
            )

    return prepared


def prepare_corpus_sentences(
    positioned_documents,
    include_unsentenced=False,
    lowercase=True,
    min_tokens=2,
):
    """
    Main M03 -> E31 preprocessing interface.
    """

    sentences = []

    for document in (
        positioned_documents
        or []
    ):
        sentences.extend(
            prepare_document_sentences(
                positioned_document=
                    document,

                include_unsentenced=
                    include_unsentenced,

                lowercase=
                    lowercase,

                min_tokens=
                    min_tokens,
            )
        )

    return sentences


# ============================================================
# 3. MODEL SELECTION
# ============================================================

def resolve_architecture(
    architecture,
):
    value = (
        str(
            architecture
        )
        .strip()
        .casefold()
        .replace(
            "_",
            "",
        )
        .replace(
            "-",
            "",
        )
    )

    cbow_aliases = {
        "cbow",
        "continuousbagofwords",
        "continuousbagofword",
    }

    skipgram_aliases = {
        "sg",
        "skipgram",
        "skipgrams",
    }

    if value in cbow_aliases:
        return {
            "name":
                "cbow",

            "gensim_sg":
                0,
        }

    if value in skipgram_aliases:
        return {
            "name":
                "skipgram",

            "gensim_sg":
                1,
        }

    raise ValueError(
        "architecture must be "
        "'cbow' or 'skipgram'."
    )


# ============================================================
# 4. GENSIM LOADER
# ============================================================

def get_word2vec_class():
    try:
        from gensim.models import (
            Word2Vec,
        )

    except ImportError as error:
        raise ImportError(
            "E31 Word2Vec requires gensim. "
            "Install it with: pip install gensim"
        ) from error

    return Word2Vec


# ============================================================
# 5. TRAIN WORD2VEC
# ============================================================

def train_word2vec(
    positioned_documents,
    architecture="skipgram",
    vector_size=100,
    window=5,
    min_count=1,
    epochs=20,
    workers=1,
    seed=42,
    include_unsentenced=False,
):
    """
    Train Word2Vec from M03 positioned_documents.

    Parameters are explicit so they can be shown in the report.

    architecture:
        "cbow"
        "skipgram"

    workers=1 is the default for reproducibility during demos.
    """

    architecture_info = (
        resolve_architecture(
            architecture
        )
    )

    sentences = (
        prepare_corpus_sentences(
            positioned_documents=
                positioned_documents,

            include_unsentenced=
                include_unsentenced,

            lowercase=
                True,

            min_tokens=
                2,
        )
    )

    if not sentences:
        raise ValueError(
            "No usable sentence tokens were found "
            "for Word2Vec training."
        )

    Word2Vec = (
        get_word2vec_class()
    )

    model = Word2Vec(
        sentences=
            sentences,

        vector_size=
            vector_size,

        window=
            window,

        min_count=
            min_count,

        workers=
            workers,

        sg=
            architecture_info[
                "gensim_sg"
            ],

        seed=
            seed,

        epochs=
            epochs,
    )

    token_count = sum(
        len(sentence)
        for sentence
        in sentences
    )

    metadata = {
        "architecture":
            architecture_info[
                "name"
            ],

        "vector_size":
            vector_size,

        "window":
            window,

        "min_count":
            min_count,

        "epochs":
            epochs,

        "workers":
            workers,

        "seed":
            seed,

        "document_count":
            len(
                positioned_documents
                or []
            ),

        "sentence_count":
            len(
                sentences
            ),

        "training_token_count":
            token_count,

        "vocabulary_size":
            len(
                model.wv
            ),

        "include_unsentenced":
            include_unsentenced,
    }

    return {
        "model":
            model,

        "metadata":
            metadata,

        "sentences":
            sentences,
    }


# ============================================================
# 6. WORD LOOKUP
# ============================================================

def has_word(
    model,
    word,
):
    word = normalize_word(
        word
    )

    return (
        bool(word)
        and word in model.wv
    )


def most_similar(
    model,
    word,
    topn=10,
):
    word = normalize_word(
        word
    )

    if not word:
        return []

    if word not in model.wv:
        return []

    return [
        {
            "word":
                candidate,

            "similarity":
                round(
                    float(
                        similarity
                    ),
                    6,
                ),
        }

        for (
            candidate,
            similarity,
        )
        in model.wv.most_similar(
            word,
            topn=
                topn,
        )
    ]


def word_similarity(
    model,
    first_word,
    second_word,
):
    first_word = normalize_word(
        first_word
    )

    second_word = normalize_word(
        second_word
    )

    if (
        first_word not in model.wv
        or second_word not in model.wv
    ):
        return None

    return round(
        float(
            model.wv.similarity(
                first_word,
                second_word,
            )
        ),
        6,
    )


# ============================================================
# 7. MODEL SAVE / LOAD
# ============================================================

def save_word2vec(
    model,
    output_path,
):
    """
    Useful when the corpus grows to hundreds or thousands of
    PubMed documents. Train once, then reuse the saved model.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        str(
            output_path
        )
    )

    return output_path


def load_word2vec(
    model_path,
):
    Word2Vec = (
        get_word2vec_class()
    )

    return Word2Vec.load(
        str(
            model_path
        )
    )


# ============================================================
# 8. TEST — PREPROCESSING INTERFACE
# ============================================================
#
# This test does not require gensim.
# It verifies M03 -> E31 sentence preparation.
# ============================================================

if __name__ == "__main__":

    positioned_documents = [
        {
            "pmid":
                "DEMO001",

            "segments":
                [
                    {
                        "segment_id":
                            0,

                        "field":
                            "abstract",

                        "words":
                            [
                                {
                                    "text": "GLP-1",
                                    "global_char_start": 0,
                                    "global_char_end": 5,
                                },
                                {
                                    "text": "improves",
                                    "global_char_start": 6,
                                    "global_char_end": 14,
                                },
                                {
                                    "text": "glucose",
                                    "global_char_start": 15,
                                    "global_char_end": 22,
                                },
                                {
                                    "text": "control",
                                    "global_char_start": 23,
                                    "global_char_end": 30,
                                },
                            ],

                        "sentences":
                            [
                                {
                                    "text":
                                        "GLP-1 improves glucose control.",

                                    "global_char_start":
                                        0,

                                    "global_char_end":
                                        31,
                                }
                            ],
                    }
                ],
        }
    ]

    sentences = (
        prepare_corpus_sentences(
            positioned_documents
        )
    )

    print(
        "=" * 70
    )

    print(
        "E31 - WORD2VEC INPUT TEST"
    )

    print(
        "=" * 70
    )

    print(
        sentences
    )

    print()

    print(
        "gensim available:",
        end=" ",
    )

    try:
        get_word2vec_class()
        print(
            "YES"
        )

    except ImportError:
        print(
            "NO"
        )
