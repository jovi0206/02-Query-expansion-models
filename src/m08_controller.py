from pathlib import Path
import tempfile

from m01_input_parser import parse_jats
from e1_01_pubmed_batch import load_pubmed_documents_batched_from_text
from e1_02_local_jats_corpus import (
    load_local_jats_corpus,
    load_prepared_corpus,
    corpus_sha256,
)
from m02_text_processor import (
    process_document,
    run_porter_corpus,
    run_preprocessing_experiment,
)
from m03_position_mapper import (
    map_document_positions,
    train_word2vec_model,
    get_word2vec_similar,
    get_word2vec_similarity,
    save_word2vec_model,
    load_word2vec_model,
)
from m04_index_builder import (
    build_index,
    analyze_zipf,
    compare_porter_zipf,
    analyze_preprocessing_zipf,
    analyze_term_statistics,
    get_document_tfidf,
)
from m05_query_processor import (
    process_query,
    correct_spelling,
    expand_semantic_query,
)
from m06_retrieval_engine import retrieve
from m07_result_processor import process_results
from e6_01_significant_words import analyze_all_conditions as _e61_analyze_all_conditions
from e8_01_domain_comparison import compare_two_domains
from project2_artifacts import load_analysis_payload, load_runtime_state
from project2_config import (
    DEFAULT_GLP1_JATS_DIR,
    PREPARED_CORPUS_PATH,
    PREPARED_ANALYSIS_PATH,
    PREPARED_RUNTIME_PATH,
    WORD2VEC_MODEL_PATH,
)


# ============================================================
# MODULE 08 — APPLICATION CONTROLLER
# ============================================================
#
# Build Flow:
#     M01 / E12 -> M02 -> M03 -> M04
#
# Search Flow:
#     M05 -> M06 -> M07
#
# Project 2 Analysis:
#     E21 / E22 / E31 / E41 / E42 / E51 / E52 / E81
#
# M08 coordinates application flow only. Algorithms remain in their
# respective modules and M09 remains presentation-only.
# ============================================================


def get_document_id(document):
    return (
        document.get("pmid")
        or document.get("pmcid")
        or document.get("filename", "")
    )


def empty_app_state():
    return {
        "index_ready": False,
        "processed_documents": [],
        "positioned_documents": [],
        "index_data": None,
        "build_errors": [],
        "pubmed_fetch_reports": [],
        "search_result": None,

        # Fixed-corpus metadata.
        "dataset_audit": {},
        "corpus_sha256": "",
        "prepared_metadata": {},
        "data_mode": "empty",

        # Project 2 results.
        "porter_result": None,
        "preprocessing_result": None,
        "zipf_result": None,
        "porter_zipf_result": None,
        "preprocessing_zipf_result": None,
        "term_statistics_result": None,
        "term_statistics_by_condition": {},
        "significant_words_by_condition": {},
        "word2vec_result": None,
        "spellcheck_result": None,
        "semantic_result": None,
        "domain_comparison_result": None,
    }


def build_from_documents(
    documents,
    dataset_audit=None,
    data_mode="documents",
):
    """Generic M01-compatible document builder: M02 -> M03 -> M04."""
    processed_documents = []
    positioned_documents = []
    errors = []
    accepted_document_ids = set()

    documents = list(documents or [])

    for document in documents:
        try:
            processed = process_document(document)
            positioned = map_document_positions(processed)

            document_id = get_document_id(positioned)
            if not document_id:
                raise ValueError("Document has no stable ID.")

            if document_id in accepted_document_ids:
                raise ValueError(
                    "Duplicate document ID detected: "
                    f"{document_id}"
                )

            accepted_document_ids.add(document_id)
            processed_documents.append(processed)
            positioned_documents.append(positioned)

        except Exception as error:
            errors.append(
                {
                    "filename": (
                        document.get("filename", "")
                        or document.get("pmid", "")
                        or "Unknown document"
                    ),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    index_data = (
        build_index(positioned_documents)
        if positioned_documents
        else None
    )

    state = empty_app_state()
    state.update(
        {
            "index_ready": index_data is not None,
            "processed_documents": processed_documents,
            "positioned_documents": positioned_documents,
            "index_data": index_data,
            "build_errors": errors,
            "dataset_audit": dict(dataset_audit or {}),
            "corpus_sha256": corpus_sha256(documents) if documents else "",
            "data_mode": data_mode,
        }
    )

    return state


# ============================================================
# INPUT PATHS
# ============================================================


def build_from_local_jats(
    jats_dir=DEFAULT_GLP1_JATS_DIR,
    target=1000,
):
    """Load the official local PubMed-selected PMC JATS corpus."""
    documents, audit = load_local_jats_corpus(
        jats_dir=jats_dir,
        target=target,
        require_pmid=True,
        require_abstract=True,
        require_english=True,
    )
    state = build_from_documents(
        documents,
        dataset_audit=audit,
        data_mode="local_jats",
    )
    return state


def build_from_prepared_corpus(
    corpus_path=PREPARED_CORPUS_PATH,
):
    """Build M02-M04 from the portable prepared abstract corpus."""
    documents, audit = load_prepared_corpus(corpus_path)
    state = build_from_documents(
        documents,
        dataset_audit=audit,
        data_mode="prepared_corpus",
    )
    return state


def get_input_name(input_file):
    return Path(getattr(input_file, "name", "input.dat")).name


def get_input_bytes(input_file):
    if hasattr(input_file, "getvalue"):
        return input_file.getvalue()
    if isinstance(input_file, (str, Path)):
        return Path(input_file).read_bytes()
    if isinstance(input_file, bytes):
        return input_file
    raise TypeError(
        "Unsupported input object. Expected UploadedFile, Path, str, or bytes."
    )


def detect_input_type(input_file):
    raw = get_input_bytes(input_file)
    stripped = raw.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"<"):
        return "xml"
    suffix = Path(get_input_name(input_file)).suffix.lower()
    if suffix == ".xml":
        return "xml"
    return "pmid_text"


def build_from_uploaded_files(
    input_files,
    pubmed_batch_size=100,
    pubmed_pause_seconds=0.35,
    pubmed_max_retries=3,
):
    """Project-1-compatible interactive input path for ad-hoc demo files."""
    documents = []
    input_errors = []
    pubmed_fetch_reports = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)

        for file_number, input_file in enumerate(input_files or [], start=1):
            input_name = get_input_name(input_file)

            try:
                input_type = detect_input_type(input_file)
                raw = get_input_bytes(input_file)

                if input_type == "xml":
                    file_folder = temp_root / f"{file_number:03d}"
                    file_folder.mkdir(parents=True, exist_ok=True)
                    xml_path = file_folder / input_name
                    xml_path.write_bytes(raw)
                    documents.append(parse_jats(xml_path))
                    continue

                readme_text = raw.decode("utf-8-sig", errors="replace")
                (
                    pubmed_documents,
                    missing_pmids,
                    fetch_report,
                ) = load_pubmed_documents_batched_from_text(
                    readme_text,
                    batch_size=pubmed_batch_size,
                    pause_seconds=pubmed_pause_seconds,
                    max_retries=pubmed_max_retries,
                    source_name=input_name,
                )

                pubmed_fetch_reports.append(fetch_report)
                for pmid in missing_pmids:
                    input_errors.append(
                        {
                            "filename": f"PMID {pmid}",
                            "error": "PubMed record was not returned by NCBI EFetch.",
                        }
                    )
                documents.extend(pubmed_documents)

            except Exception as error:
                input_errors.append(
                    {
                        "filename": input_name,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    app_state = build_from_documents(
        documents,
        data_mode="uploaded_files",
    )
    app_state["build_errors"] = input_errors + app_state["build_errors"]
    app_state["pubmed_fetch_reports"] = pubmed_fetch_reports
    return app_state


# ============================================================
# BASELINE RETRIEVAL
# ============================================================


def run_search(app_state, query, query_type="auto"):
    _require_index(app_state)

    query_plan = process_query(query=query, query_type=query_type)
    retrieval_data = retrieve(
        index_data=app_state["index_data"],
        positioned_documents=app_state["positioned_documents"],
        query_plan=query_plan,
    )
    search_result = process_results(
        retrieval_data=retrieval_data,
        positioned_documents=app_state["positioned_documents"],
    )
    app_state["search_result"] = search_result
    return search_result


# ============================================================
# PROJECT 2 ANALYSIS CONTROLLER
# ============================================================


def _require_app_state(app_state):
    if not app_state:
        raise ValueError("app_state is required.")


def _require_index(app_state):
    _require_app_state(app_state)
    if not app_state.get("index_ready", False):
        raise ValueError("Index is not ready. Build or load documents first.")


def run_porter_analysis(app_state):
    _require_app_state(app_state)
    processed_documents = app_state.get("processed_documents", [])
    if not processed_documents:
        raise ValueError("No processed documents are available for Porter analysis.")

    result = run_porter_corpus(processed_documents)
    app_state["porter_result"] = result
    return result


def run_preprocessing_analysis(app_state, top_n=50, stopwords=None):
    _require_app_state(app_state)
    processed_documents = app_state.get("processed_documents", [])
    if not processed_documents:
        raise ValueError("No processed documents are available for preprocessing analysis.")

    result = run_preprocessing_experiment(
        processed_documents=processed_documents,
        top_n=top_n,
        stopwords=stopwords,
    )
    app_state["preprocessing_result"] = result
    app_state["preprocessing_zipf_result"] = None
    app_state["term_statistics_result"] = None
    app_state["term_statistics_by_condition"] = {}
    return result


def run_zipf_analysis(app_state, top_n=None):
    _require_index(app_state)
    result = analyze_zipf(index_data=app_state["index_data"], top_n=top_n)
    app_state["zipf_result"] = result
    return result


def run_porter_zipf_analysis(app_state, top_n=None):
    _require_app_state(app_state)
    porter_result = app_state.get("porter_result")
    if porter_result is None:
        porter_result = run_porter_analysis(app_state)

    result = compare_porter_zipf(porter_result=porter_result, top_n=top_n)
    app_state["porter_zipf_result"] = result
    return result


def run_preprocessing_zipf_analysis(app_state, top_n=None):
    _require_app_state(app_state)
    preprocessing_result = app_state.get("preprocessing_result")
    if preprocessing_result is None:
        preprocessing_result = run_preprocessing_analysis(app_state)

    result = analyze_preprocessing_zipf(
        preprocessing_result=preprocessing_result,
        top_n=top_n,
    )
    app_state["preprocessing_zipf_result"] = result
    return result


def run_term_statistics_analysis(
    app_state,
    terms=None,
    condition_key=None,
    cf_df_top_n=20,
    idf_top_n=10,
    log_base=10,
):
    _require_index(app_state)

    condition_result = None
    if condition_key is not None:
        preprocessing_result = app_state.get("preprocessing_result")
        if preprocessing_result is None:
            preprocessing_result = run_preprocessing_analysis(app_state)

        condition_result = preprocessing_result.get("conditions", {}).get(condition_key)
        if condition_result is None:
            raise ValueError(f"Unknown preprocessing condition: {condition_key}")

    result = analyze_term_statistics(
        index_data=None if condition_result is not None else app_state["index_data"],
        condition_result=condition_result,
        terms=terms,
        cf_df_top_n=cf_df_top_n,
        idf_top_n=idf_top_n,
        log_base=log_base,
    )
    result["condition_key"] = condition_key
    app_state["term_statistics_result"] = result

    key = condition_key or "m04_index"
    app_state.setdefault("term_statistics_by_condition", {})[key] = result
    return result


def run_all_term_statistics(app_state, cf_df_top_n=50, idf_top_n=20, log_base=10):
    """Prepare CF/DF/IDF tables for every A-D preprocessing condition."""
    preprocessing = app_state.get("preprocessing_result")
    if preprocessing is None:
        preprocessing = run_preprocessing_analysis(app_state, top_n=max(50, cf_df_top_n))

    output = {}
    for condition_key in preprocessing.get("condition_order", []):
        output[condition_key] = run_term_statistics_analysis(
            app_state,
            condition_key=condition_key,
            cf_df_top_n=cf_df_top_n,
            idf_top_n=idf_top_n,
            log_base=log_base,
        )

    app_state["term_statistics_by_condition"] = output
    if output:
        # Keep one convenient current result; UI can switch among all four.
        default_key = "B_no_punctuation" if "B_no_punctuation" in output else next(iter(output))
        app_state["term_statistics_result"] = output[default_key]
    return output


def run_significant_words_analysis(app_state, top_n=20):
    """
    Build the V3 exploratory Significant Words analysis for A-D.

    This does NOT invent fixed upper/lower cut-off ranks.  It uses the
    transparent E61 candidate proxy and leaves the classical cut-offs
    explicitly unresolved unless the course later supplies a rule.
    """
    _require_app_state(app_state)

    term_stats = app_state.get("term_statistics_by_condition") or {}
    if not term_stats:
        term_stats = run_all_term_statistics(
            app_state,
            cf_df_top_n=50,
            idf_top_n=20,
            log_base=10,
        )

    result = _e61_analyze_all_conditions(
        term_stats,
        top_n=top_n,
    )
    app_state["significant_words_by_condition"] = result
    return result


def run_document_tfidf_analysis(
    app_state,
    document_id,
    terms=None,
    log_base=10,
    normalize_tf=False,
    top_n=None,
):
    _require_index(app_state)
    return get_document_tfidf(
        index_data=app_state["index_data"],
        document_id=document_id,
        terms=terms,
        log_base=log_base,
        normalize_tf=normalize_tf,
        top_n=top_n,
    )


def train_word2vec_analysis(
    app_state,
    architecture="skipgram",
    vector_size=100,
    window=5,
    min_count=2,
    epochs=20,
    workers=1,
    seed=42,
    include_unsentenced=False,
):
    _require_app_state(app_state)
    positioned_documents = app_state.get("positioned_documents", [])
    if not positioned_documents:
        raise ValueError("No positioned documents are available for Word2Vec training.")

    result = train_word2vec_model(
        positioned_documents=positioned_documents,
        architecture=architecture,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=workers,
        seed=seed,
        include_unsentenced=include_unsentenced,
    )
    app_state["word2vec_result"] = result
    return result


def save_current_word2vec(app_state, output_path=WORD2VEC_MODEL_PATH):
    model = _require_word2vec_model(app_state)
    return save_word2vec_model(model, output_path)


def load_saved_word2vec(app_state, model_path=WORD2VEC_MODEL_PATH, metadata=None):
    _require_app_state(app_state)
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Word2Vec model not found: {model_path}")

    model = load_word2vec_model(model_path)
    app_state["word2vec_result"] = {
        "model": model,
        "metadata": dict(metadata or {}),
        "sentences": [],
        "loaded_from": str(model_path),
    }
    return app_state["word2vec_result"]


def _require_word2vec_model(app_state):
    _require_app_state(app_state)
    word2vec_result = app_state.get("word2vec_result")
    if not word2vec_result or word2vec_result.get("model") is None:
        raise ValueError("Word2Vec model is not ready. Train or load E31 first.")
    return word2vec_result["model"]


def run_word2vec_similar(app_state, word, topn=10):
    model = _require_word2vec_model(app_state)
    return get_word2vec_similar(model=model, word=word, topn=topn)


def run_word2vec_similarity(app_state, first_word, second_word):
    model = _require_word2vec_model(app_state)
    return get_word2vec_similarity(
        model=model,
        first_word=first_word,
        second_word=second_word,
    )


def run_spelling_correction(app_state, query, max_distance=2, top_n=5):
    _require_index(app_state)
    result = correct_spelling(
        query=query,
        index_data=app_state["index_data"],
        max_distance=max_distance,
        top_n=top_n,
    )
    app_state["spellcheck_result"] = result
    return result


def run_semantic_expansion(app_state, query, top_n_per_word=3, min_similarity=0.5):
    model = _require_word2vec_model(app_state)
    result = expand_semantic_query(
        query=query,
        model=model,
        top_n_per_word=top_n_per_word,
        min_similarity=min_similarity,
    )
    app_state["semantic_result"] = result
    return result


def run_optional_domain_comparison(
    app_state,
    other_documents,
    label_a="GLP-1 Medical",
    label_b="Domain B",
    condition_key="C_no_stopwords",
    sample_size=500,
    top_n=10,
):
    """Optional challenge using the same analyzer for both domains."""
    _require_app_state(app_state)

    first = list(app_state.get("processed_documents", []))[: int(sample_size)]
    other_state = build_from_documents(list(other_documents or [])[: int(sample_size)])
    second = other_state.get("processed_documents", [])

    result = compare_two_domains(
        first,
        second,
        label_a=label_a,
        label_b=label_b,
        condition_key=condition_key,
        top_n=top_n,
    )
    app_state["domain_comparison_result"] = result
    return result


def run_optional_domain_comparison_from_local_jats(
    app_state,
    other_jats_dir,
    label_b="Domain B",
    condition_key="C_no_stopwords",
    sample_size=500,
    top_n=10,
):
    other_documents, other_audit = load_local_jats_corpus(
        jats_dir=other_jats_dir,
        target=sample_size,
        require_pmid=True,
        require_abstract=True,
        require_english=True,
    )
    result = run_optional_domain_comparison(
        app_state,
        other_documents,
        label_a="GLP-1 Medical",
        label_b=label_b,
        condition_key=condition_key,
        sample_size=sample_size,
        top_n=top_n,
    )
    result["domain_b_audit"] = other_audit
    return result


# ============================================================
# RUN-EVERYTHING / PREPARED RESULTS
# ============================================================


def run_required_analysis_suite(app_state):
    """
    Run every deterministic required analysis used by the report/UI.

    Word2Vec is intentionally separate because it is the expensive model step.
    """
    _require_index(app_state)

    run_preprocessing_analysis(app_state, top_n=50)
    run_zipf_analysis(app_state, top_n=None)
    run_preprocessing_zipf_analysis(app_state, top_n=None)
    run_all_term_statistics(app_state, cf_df_top_n=50, idf_top_n=20, log_base=10)
    run_significant_words_analysis(app_state, top_n=20)
    run_porter_analysis(app_state)
    run_porter_zipf_analysis(app_state, top_n=None)

    return {
        "preprocessing_result": app_state.get("preprocessing_result"),
        "zipf_result": app_state.get("zipf_result"),
        "preprocessing_zipf_result": app_state.get("preprocessing_zipf_result"),
        "term_statistics_by_condition": app_state.get("term_statistics_by_condition", {}),
        "significant_words_by_condition": app_state.get("significant_words_by_condition", {}),
        "porter_result": app_state.get("porter_result"),
        "porter_zipf_result": app_state.get("porter_zipf_result"),
    }


def build_prepared_analysis_payload(app_state):
    """Return JSON-safe analysis results; excludes the in-memory gensim model."""
    _require_app_state(app_state)

    word2vec_result = app_state.get("word2vec_result") or {}
    word2vec_metadata = dict(word2vec_result.get("metadata") or {})

    porter_result = app_state.get("porter_result") or None
    compact_porter = porter_result
    if porter_result:
        # The per-document original/stemmed token arrays are very large and are
        # not needed for prepared report figures. Keep corpus-level frequencies
        # and summary numbers; live Porter can still be rerun from the corpus.
        compact_porter = {
            key: value
            for key, value in porter_result.items()
            if key != "documents"
        }

    return {
        "dataset_audit": app_state.get("dataset_audit", {}),
        "corpus_sha256": app_state.get("corpus_sha256", ""),
        "idf_convention": "log10(N / DF), no smoothing",
        "index_summary": get_index_summary(app_state),
        "preprocessing_result": app_state.get("preprocessing_result"),
        "zipf_result": app_state.get("zipf_result"),
        "preprocessing_zipf_result": app_state.get("preprocessing_zipf_result"),
        "term_statistics_by_condition": app_state.get("term_statistics_by_condition", {}),
        "significant_words_by_condition": app_state.get("significant_words_by_condition", {}),
        "porter_result": compact_porter,
        "porter_zipf_result": app_state.get("porter_zipf_result"),
        "word2vec_metadata": word2vec_metadata,
        "domain_comparison_result": app_state.get("domain_comparison_result"),
    }


def apply_prepared_analysis_payload(app_state, payload):
    _require_app_state(app_state)
    payload = payload or {}

    # Verify that prepared results belong to this exact corpus.
    payload_hash = payload.get("corpus_sha256", "")
    state_hash = app_state.get("corpus_sha256", "")
    if payload_hash and state_hash and payload_hash != state_hash:
        raise ValueError(
            "Prepared analysis corpus hash does not match the loaded corpus. "
            "Rebuild artifacts before demo/deployment."
        )

    for key in (
        "preprocessing_result",
        "zipf_result",
        "preprocessing_zipf_result",
        "term_statistics_by_condition",
        "significant_words_by_condition",
        "porter_result",
        "porter_zipf_result",
        "domain_comparison_result",
    ):
        if key in payload:
            app_state[key] = payload.get(key)

    term_stats = app_state.get("term_statistics_by_condition") or {}
    if term_stats:
        default_key = "B_no_punctuation" if "B_no_punctuation" in term_stats else next(iter(term_stats))
        app_state["term_statistics_result"] = term_stats[default_key]

    return app_state


def load_prepared_project_state(
    corpus_path=PREPARED_CORPUS_PATH,
    analysis_path=PREPARED_ANALYSIS_PATH,
    runtime_path=PREPARED_RUNTIME_PATH,
    word2vec_model_path=WORD2VEC_MODEL_PATH,
):
    """
    Preferred demo/cloud path.

    V3 first loads the prepared M02/M03/M04 runtime cache so M09 does
    not rebuild the 1,000-document index on every cold start.  If the
    runtime cache is absent, it safely falls back to the V2 behavior of
    rebuilding from the portable prepared corpus.
    """
    runtime_metadata = {}

    if Path(runtime_path).exists():
        runtime, runtime_metadata = load_runtime_state(runtime_path)
        state = empty_app_state()
        state.update(
            {
                "index_ready": runtime.get("index_data") is not None,
                "processed_documents": runtime.get("processed_documents", []),
                "positioned_documents": runtime.get("positioned_documents", []),
                "index_data": runtime.get("index_data"),
                "build_errors": runtime.get("build_errors", []),
                "dataset_audit": runtime.get("dataset_audit", {}),
                "corpus_sha256": runtime.get("corpus_sha256", ""),
                "data_mode": "prepared_runtime",
            }
        )
    else:
        state = build_from_prepared_corpus(corpus_path)

    if Path(analysis_path).exists():
        payload, metadata = load_analysis_payload(analysis_path)
        apply_prepared_analysis_payload(state, payload)
        state["prepared_metadata"] = dict(metadata or {})
        if runtime_metadata:
            state["prepared_metadata"]["runtime"] = runtime_metadata

        model_metadata = payload.get("word2vec_metadata") or {}
        if Path(word2vec_model_path).exists():
            try:
                load_saved_word2vec(
                    state,
                    model_path=word2vec_model_path,
                    metadata=model_metadata,
                )
            except Exception as error:
                state["build_errors"].append(
                    {
                        "filename": str(word2vec_model_path),
                        "error": f"Word2Vec load: {type(error).__name__}: {error}",
                    }
                )

    state["data_mode"] = (
        "prepared_demo_cache"
        if Path(runtime_path).exists()
        else "prepared_project"
    )
    return state


# ============================================================
# STATUS / REPORT HELPERS
# ============================================================


def get_extension_status(app_state):
    state = app_state or {}
    return {
        "porter_ready": state.get("porter_result") is not None,
        "preprocessing_ready": state.get("preprocessing_result") is not None,
        "zipf_ready": state.get("zipf_result") is not None,
        "porter_zipf_ready": state.get("porter_zipf_result") is not None,
        "preprocessing_zipf_ready": state.get("preprocessing_zipf_result") is not None,
        "term_statistics_ready": bool(state.get("term_statistics_by_condition")),
        "significant_words_ready": bool(state.get("significant_words_by_condition")),
        "word2vec_ready": bool(
            state.get("word2vec_result")
            and state.get("word2vec_result", {}).get("model") is not None
        ),
        "spellcheck_ready": state.get("spellcheck_result") is not None,
        "semantic_ready": state.get("semantic_result") is not None,
        "optional_domain_ready": state.get("domain_comparison_result") is not None,
    }


def get_index_summary(app_state):
    if not app_state or not app_state.get("index_ready", False):
        return {
            "documents": 0,
            "unique_terms": 0,
            "total_terms": 0,
            "average_terms_per_document": 0.0,
            "errors": len((app_state or {}).get("build_errors", [])),
        }

    index_data = app_state["index_data"]
    documents = index_data.get(
        "total_documents",
        len(app_state.get("processed_documents", [])),
    )
    total_terms = index_data.get("total_terms", 0)

    return {
        "documents": documents,
        "unique_terms": index_data.get("unique_terms", len(index_data.get("index", {}))),
        "total_terms": total_terms,
        "average_terms_per_document": (
            total_terms / documents if documents else 0.0
        ),
        "errors": len(app_state.get("build_errors", [])),
    }


def get_document_stats(app_state):
    rows = []
    if not app_state:
        return rows

    for document in app_state.get("processed_documents", []):
        rows.append(
            {
                "PMID": document.get("pmid", ""),
                "PMCID": document.get("pmcid", ""),
                "Title": document.get("title", ""),
                "File": document.get("filename", ""),
                "Format": document.get("source_format", "Unknown"),
                "Characters": document.get("character_count", 0),
                "Words": document.get("computed_word_count", 0),
                "Sentences": document.get("sentence_count", 0),
            }
        )
    return rows


def get_project_paths():
    return {
        "raw_glp1_jats_dir": str(DEFAULT_GLP1_JATS_DIR),
        "raw_glp1_jats_exists": Path(DEFAULT_GLP1_JATS_DIR).exists(),
        "prepared_corpus": str(PREPARED_CORPUS_PATH),
        "prepared_corpus_exists": Path(PREPARED_CORPUS_PATH).exists(),
        "prepared_analysis": str(PREPARED_ANALYSIS_PATH),
        "prepared_analysis_exists": Path(PREPARED_ANALYSIS_PATH).exists(),
        "prepared_runtime": str(PREPARED_RUNTIME_PATH),
        "prepared_runtime_exists": Path(PREPARED_RUNTIME_PATH).exists(),
        "word2vec_model": str(WORD2VEC_MODEL_PATH),
        "word2vec_model_exists": Path(WORD2VEC_MODEL_PATH).exists(),
    }


def prepared_project_available():
    return Path(PREPARED_CORPUS_PATH).exists()
