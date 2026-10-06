import argparse
import json
import time
from pathlib import Path

from e1_02_local_jats_corpus import (
    load_local_jats_corpus,
    save_prepared_corpus,
    audit_glp1_representations,
)
from m08_controller import (
    build_from_documents,
    run_required_analysis_suite,
    train_word2vec_analysis,
    save_current_word2vec,
    run_optional_domain_comparison,
    build_prepared_analysis_payload,
    get_index_summary,
)
from project2_artifacts import save_analysis_payload, save_manifest, save_runtime_state
from project2_config import (
    DEFAULT_GLP1_JATS_DIR,
    PREPARED_CORPUS_PATH,
    PREPARED_ANALYSIS_PATH,
    PREPARED_MANIFEST_PATH,
    PREPARED_RUNTIME_PATH,
    WORD2VEC_MODEL_PATH,
    GLP1_V2_AUDIT_PATH,
    OPTIONAL_DOMAIN_B_JATS_DIR,
    ensure_output_directories,
)


# ============================================================
# PROJECT 2 — ONE-COMMAND PREPARATION PIPELINE
# ============================================================
#
# Default run:
#     python src\project2_prepare_all.py
#
# Input:
#     Project2/data/glp1_pubmed_jats_v2/valid_1200/*.xml
#
# The official loader freezes the first 1,000 documents it accepts in
# original PubMed relevance order, guaranteeing an actual 1,000-document
# experiment corpus.
#
# Output (inside the app/repository folder):
#     prepared_data/glp1_1000_abstracts.jsonl.gz
#     artifacts/project2_analysis.json.gz
#     artifacts/project2_manifest.json
#     artifacts/glp1_word2vec.model
#
# The generated corpus/results are what M09 can load on both local PC and
# Streamlit Cloud, so the report numbers remain identical.
# ============================================================


def _step(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_GLP1_JATS_DIR),
        help="Directory containing the official GLP-1 JATS XML files.",
    )
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--skip-word2vec", action="store_true")
    parser.add_argument(
        "--domain-b-dir",
        default=str(OPTIONAL_DOMAIN_B_JATS_DIR),
        help="Optional second-domain JATS folder (500 docs recommended).",
    )
    parser.add_argument("--skip-optional", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_directories()
    total_started = time.perf_counter()

    _step("STEP 1/6 — Load and audit official GLP-1 JATS corpus")
    documents, audit = load_local_jats_corpus(
        jats_dir=Path(args.raw_dir),
        target=args.target,
        require_pmid=True,
        require_abstract=True,
        require_english=True,
    )

    # V3: preserve the corrected downloader audit in the prepared corpus so
    # local/cloud M09 can display the 1,200 validated pool and GLP-1
    # representation coverage without depending on the raw data folder.
    if Path(GLP1_V2_AUDIT_PATH).exists():
        try:
            source_audit = json.loads(Path(GLP1_V2_AUDIT_PATH).read_text(encoding="utf-8"))
            audit["source_collection_audit"] = source_audit
            audit["source_valid_pool_documents"] = source_audit.get("valid_pool_documents")
            audit["source_formal_documents"] = source_audit.get("formal_documents")
            audit["source_glp1_any_form_documents"] = source_audit.get(
                "any_requested_or_extra_form_unique_documents"
            )
            audit["source_mapping_rule"] = source_audit.get("mapping_rule", "")
        except Exception as error:
            audit.setdefault("source_audit_warnings", []).append(
                f"{type(error).__name__}: {error}"
            )

    print("XML files found      :", audit.get("xml_files_found", 0))
    print("Accepted documents   :", audit.get("accepted_documents", 0))
    print("Unique PMIDs         :", audit.get("unique_pmids", 0))
    print("Missing PMID         :", len(audit.get("missing_pmid", [])))
    print("Missing abstract     :", len(audit.get("missing_abstract", [])))
    print("Duplicate PMID       :", len(audit.get("duplicate_pmids", [])))
    print("Non-English excluded :", len(audit.get("non_english", [])))
    print("Parse errors         :", len(audit.get("parse_errors", [])))

    if len(documents) == 0:
        raise RuntimeError("No valid documents were loaded.")

    if args.target and len(documents) < args.target:
        raise RuntimeError(
            f"Target corpus requires {args.target} system-accepted documents, "
            f"but only {len(documents)} were accepted from the reserve pool."
        )

    # Re-audit GLP-1 representation on the exact same documents used by all
    # downstream experiments; never mix downloader audit denominators with
    # the final system corpus.
    system_glp1_audit = audit_glp1_representations(documents)
    audit["system_glp1_representation_audit"] = system_glp1_audit
    audit["system_glp1_any_form_documents"] = system_glp1_audit.get(
        "any_form_unique_documents", 0
    )
    audit["system_glp1_total_documents"] = system_glp1_audit.get(
        "total_documents", len(documents)
    )
    audit["system_glp1_any_form_coverage"] = system_glp1_audit.get(
        "any_form_coverage", 0.0
    )
    print(
        "GLP-1 audited forms   :",
        f"{audit['system_glp1_any_form_documents']}/"
        f"{audit['system_glp1_total_documents']}",
        f"({audit['system_glp1_any_form_coverage']:.1%})",
    )

    _step("STEP 2/6 — Save portable fixed abstract corpus")
    corpus_path = save_prepared_corpus(
        documents,
        audit,
        output_path=PREPARED_CORPUS_PATH,
    )
    print("Prepared corpus:", corpus_path)

    _step("STEP 3/6 — Build M02 -> M03 -> M04")
    build_started = time.perf_counter()
    app_state = build_from_documents(
        documents,
        dataset_audit=audit,
        data_mode="prepare_all",
    )
    build_seconds = time.perf_counter() - build_started
    summary = get_index_summary(app_state)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Build time: {build_seconds:.2f} s")

    runtime_path = save_runtime_state(app_state, PREPARED_RUNTIME_PATH)
    print("Prepared runtime cache:", runtime_path)

    if not app_state.get("index_ready"):
        raise RuntimeError("M04 index build failed.")

    _step("STEP 4/6 — Run all required deterministic Project 2 analyses")
    analysis_started = time.perf_counter()
    run_required_analysis_suite(app_state)
    analysis_seconds = time.perf_counter() - analysis_started
    print(f"Required analysis time: {analysis_seconds:.2f} s")

    comparison = (
        app_state.get("preprocessing_zipf_result", {})
        .get("comparison", [])
    )
    if comparison:
        print()
        print("A-D regression summary")
        for row in comparison:
            print(
                f"{row.get('condition'):22s} "
                f"tokens={row.get('total_tokens', 0):8d} "
                f"vocab={row.get('vocabulary_size', 0):7d} "
                f"k={row.get('zipf_exponent')!s:>10} "
                f"R2={row.get('r_squared')!s:>10} "
                f"RMSE={row.get('rmse')!s:>10}"
            )

    _step("STEP 5/6 — Word2Vec and optional two-domain challenge")
    word2vec_seconds = None
    if args.skip_word2vec:
        print("Word2Vec skipped by command-line option.")
    else:
        w2v_started = time.perf_counter()
        result = train_word2vec_analysis(
            app_state,
            architecture="skipgram",
            vector_size=100,
            window=5,
            min_count=2,
            epochs=20,
            workers=1,
            seed=42,
            include_unsentenced=False,
        )
        save_current_word2vec(app_state, WORD2VEC_MODEL_PATH)
        word2vec_seconds = time.perf_counter() - w2v_started
        print("Word2Vec model:", WORD2VEC_MODEL_PATH)
        print("Word2Vec metadata:")
        print(json.dumps(result.get("metadata", {}), indent=2, ensure_ascii=False))
        print(f"Word2Vec time: {word2vec_seconds:.2f} s")

    optional_status = "not_run"
    domain_b_dir = Path(args.domain_b_dir)
    if not args.skip_optional and domain_b_dir.exists():
        try:
            domain_b_docs, domain_b_audit = load_local_jats_corpus(
                jats_dir=domain_b_dir,
                target=500,
                require_pmid=True,
                require_abstract=True,
                require_english=True,
            )
            if domain_b_docs:
                run_optional_domain_comparison(
                    app_state,
                    domain_b_docs,
                    label_a="GLP-1 Medical",
                    label_b=domain_b_dir.name,
                    condition_key="C_no_stopwords",
                    sample_size=500,
                    top_n=10,
                )
                optional_status = "completed"
                print("Optional two-domain comparison: completed")
                print("Domain B accepted:", domain_b_audit.get("accepted_documents", 0))
            else:
                optional_status = "domain_b_empty"
        except Exception as error:
            optional_status = f"error: {type(error).__name__}: {error}"
            print("Optional challenge warning:", optional_status)
    else:
        print("Optional Domain B folder not present; system support remains ready.")

    _step("STEP 6/6 — Save prepared report/demo artifacts")
    payload = build_prepared_analysis_payload(app_state)
    analysis_path = save_analysis_payload(payload, PREPARED_ANALYSIS_PATH)

    total_seconds = time.perf_counter() - total_started
    manifest = {
        "project": "NCKU Biomedical Information Retrieval Project 2",
        "official_corpus": "GLP-1 PubMed-selected -> direct PMC JATS (pubmed_pmc)",
        "analysis_scope": "main abstract body only; section headings excluded",
        "idf_convention": "log10(N / DF), no smoothing",
        "official_raw_directory": str(Path(args.raw_dir)),
        "corpus_selection_rule": (
            "first 1000 documents accepted by the official Project 2 loader "
            "from the validated 1200-document reserve pool, preserving PubMed relevance order"
        ),
        "prepared_runtime": str(PREPARED_RUNTIME_PATH),
        "documents": len(documents),
        "corpus_sha256": app_state.get("corpus_sha256", ""),
        "raw_jats_dir": str(Path(args.raw_dir)),
        "prepared_corpus": str(PREPARED_CORPUS_PATH),
        "prepared_analysis": str(PREPARED_ANALYSIS_PATH),
        "word2vec_model": (
            str(WORD2VEC_MODEL_PATH)
            if WORD2VEC_MODEL_PATH.exists()
            else None
        ),
        "optional_domain_status": optional_status,
        "timings_seconds": {
            "build": build_seconds,
            "required_analysis": analysis_seconds,
            "word2vec": word2vec_seconds,
            "total": total_seconds,
        },
        "index_summary": summary,
        "dataset_audit": audit,
    }
    manifest_path = save_manifest(manifest, PREPARED_MANIFEST_PATH)

    print("Prepared analysis:", analysis_path)
    print("Prepared runtime :", PREPARED_RUNTIME_PATH)
    print("Manifest         :", manifest_path)
    print("Corpus SHA-256   :", app_state.get("corpus_sha256", ""))
    print(f"TOTAL TIME       : {total_seconds:.2f} s")
    print()
    print("PROJECT 2 PREPARATION: PASS")


if __name__ == "__main__":
    main()
