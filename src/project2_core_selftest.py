"""Project 2 V3 core extension smoke test.

Run from project root:
    python src/project2_core_selftest.py

This test does not call PubMed or modify project data.
"""

import math

from m08_controller import (
    run_preprocessing_analysis,
    run_preprocessing_zipf_analysis,
    run_term_statistics_analysis,
    run_significant_words_analysis,
    get_extension_status,
)


def main():
    processed_documents = [
        {
            "pmid": "TEST1",
            "search_segments": [
                {"text": "The data, data. cancer treatments improve outcomes."},
            ],
        },
        {
            "pmid": "TEST2",
            "search_segments": [
                {"text": "Cancer treatment and the patient data were analyzed."},
            ],
        },
    ]

    index_data = {
        "total_documents": 2,
        "total_terms": 10,
        "unique_terms": 4,
        "documents": {
            "TEST1": {"total_search_words": 5},
            "TEST2": {"total_search_words": 5},
        },
        "index": {
            "cancer": {
                "total_frequency": 2,
                "document_frequency": 2,
                "postings": {"TEST1": [{}], "TEST2": [{}]},
            },
            "data": {
                "total_frequency": 3,
                "document_frequency": 2,
                "postings": {"TEST1": [{}, {}], "TEST2": [{}]},
            },
            "treatment": {
                "total_frequency": 1,
                "document_frequency": 1,
                "postings": {"TEST2": [{}]},
            },
            "rare": {
                "total_frequency": 2,
                "document_frequency": 1,
                "postings": {"TEST1": [{}, {}]},
            },
        },
    }

    app_state = {
        "index_ready": True,
        "processed_documents": processed_documents,
        "positioned_documents": [],
        "index_data": index_data,
        "build_errors": [],
        "pubmed_fetch_reports": [],
        "search_result": None,
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
    }

    preprocessing = run_preprocessing_analysis(app_state)
    assert len(preprocessing["conditions"]) == 4

    zipf = run_preprocessing_zipf_analysis(app_state)
    assert len(zipf["comparison"]) == 4

    stats = run_term_statistics_analysis(app_state)
    assert stats["cf_df_rows"]
    assert stats["idf_rows"]
    assert stats["log_base"] == 10
    rare = next(row for row in stats["rows"] if row["term"] == "rare")
    assert math.isclose(rare["idf"], math.log10(2 / 1), rel_tol=1e-12)

    significant = run_significant_words_analysis(app_state, top_n=3)
    assert significant

    status = get_extension_status(app_state)
    assert status["preprocessing_ready"]
    assert status["preprocessing_zipf_ready"]
    assert status["term_statistics_ready"]
    assert status["significant_words_ready"]

    print("Project 2 V3 core self-test: PASS")
    print("Preprocessing conditions:", preprocessing["condition_order"])
    print("Regression rows:", len(zipf["comparison"]))
    print("CF/DF rows:", len(stats["cf_df_rows"]))
    print("IDF rows:", len(stats["idf_rows"]), "| log base:", stats["log_base"])
    print("Significant-word analyses:", len(significant))


if __name__ == "__main__":
    main()
