from pathlib import Path


# ============================================================
# PROJECT 2 — CENTRAL PATH CONFIGURATION (V3)
# ============================================================
#
# Expected local layout:
#
# Project2/
# ├─ data/
# │  └─ glp1_pubmed_jats_v2/
# │     ├─ valid_1200/             <- validated reserve pool
# │     ├─ formal_1000/            <- OFFICIAL experiment corpus
# │     └─ reports/
# │        └─ audit_summary.json
# └─ 02-query expansion models/
#    ├─ src/
#    ├─ prepared_data/             <- generated portable corpus
#    └─ artifacts/                 <- generated prepared analyses/runtime
#
# V3 switches the official raw corpus to the corrected direct
# PubMed -> PMC mapping dataset.  The original corpus is not deleted.
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
APP_ROOT = SRC_DIR.parent
PROJECT2_ROOT = APP_ROOT.parent
RAW_DATA_ROOT = PROJECT2_ROOT / "data"

# Corrected V2 source corpus.
#
# The downloader's formal_1000 folder was validated with the downloader's
# abstract rule. The official Project 2 loader is slightly stricter, so a few
# documents can be rejected later. To guarantee that the actual experiment
# always contains exactly 1,000 system-accepted documents, V3 reads from the
# validated 1,200-document reserve pool and freezes the first 1,000 documents
# accepted by the official Project 2 loader in PubMed relevance order.
GLP1_V2_ROOT = RAW_DATA_ROOT / "glp1_pubmed_jats_v2"
GLP1_V2_VALID_POOL_DIR = GLP1_V2_ROOT / "valid_1200"
DEFAULT_GLP1_JATS_DIR = GLP1_V2_VALID_POOL_DIR
GLP1_V2_AUDIT_PATH = GLP1_V2_ROOT / "reports" / "audit_summary.json"
GLP1_V2_POOL_MANIFEST_PATH = GLP1_V2_ROOT / "reports" / "valid_1200_manifest.csv"
GLP1_V2_FORMAL_MANIFEST_PATH = GLP1_V2_ROOT / "reports" / "formal_1000_manifest.csv"

PREPARED_DATA_DIR = APP_ROOT / "prepared_data"
ARTIFACT_DIR = APP_ROOT / "artifacts"

PREPARED_CORPUS_PATH = PREPARED_DATA_DIR / "glp1_1000_abstracts.jsonl.gz"
PREPARED_ANALYSIS_PATH = ARTIFACT_DIR / "project2_analysis.json.gz"
PREPARED_MANIFEST_PATH = ARTIFACT_DIR / "project2_manifest.json"
PREPARED_RUNTIME_PATH = ARTIFACT_DIR / "project2_runtime.pkl.gz"
WORD2VEC_MODEL_PATH = ARTIFACT_DIR / "glp1_word2vec.model"

# Optional challenge defaults. Domain A can reuse the first 500
# documents of the official GLP-1 corpus. Domain B is intentionally
# generic so a second corpus can be added later without changing code.
OPTIONAL_DOMAIN_B_JATS_DIR = RAW_DATA_ROOT / "optional_domain_b_jats_xml"
OPTIONAL_DOMAIN_B_PREPARED_PATH = PREPARED_DATA_DIR / "optional_domain_b.jsonl.gz"


def ensure_output_directories():
    PREPARED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("SRC_DIR                       :", SRC_DIR)
    print("APP_ROOT                      :", APP_ROOT)
    print("PROJECT2_ROOT                 :", PROJECT2_ROOT)
    print("DEFAULT_GLP1_JATS_DIR         :", DEFAULT_GLP1_JATS_DIR)
    print("GLP1_V2_VALID_POOL_DIR         :", GLP1_V2_VALID_POOL_DIR)
    print("GLP1_V2_AUDIT_PATH            :", GLP1_V2_AUDIT_PATH)
    print("PREPARED_CORPUS_PATH          :", PREPARED_CORPUS_PATH)
    print("PREPARED_ANALYSIS_PATH        :", PREPARED_ANALYSIS_PATH)
    print("PREPARED_RUNTIME_PATH         :", PREPARED_RUNTIME_PATH)
    print("WORD2VEC_MODEL_PATH           :", WORD2VEC_MODEL_PATH)
