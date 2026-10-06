import gzip
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from project2_config import (
    PREPARED_ANALYSIS_PATH,
    PREPARED_MANIFEST_PATH,
    PREPARED_RUNTIME_PATH,
    ensure_output_directories,
)


# ============================================================
# PROJECT 2 — PREPARED RESULT ARTIFACTS (V3)
# ============================================================
#
# JSON/gzip:
#     report/analysis outputs that are inspectable and portable.
#
# Pickle/gzip runtime cache:
#     trusted, locally generated M02->M04 runtime structures used only
#     to avoid rebuilding the 1,000-document index every time M09 starts.
#     It is never accepted from an arbitrary user upload.
# ============================================================


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_analysis_payload(payload, output_path=PREPARED_ANALYSIS_PATH):
    ensure_output_directories()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapped = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }

    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(
            wrapped,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )

    return output_path


def load_analysis_payload(input_path=PREPARED_ANALYSIS_PATH):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Prepared analysis not found: {input_path}")

    with gzip.open(input_path, "rt", encoding="utf-8") as handle:
        wrapped = json.load(handle)

    return wrapped.get("payload", {}), {
        "schema_version": wrapped.get("schema_version"),
        "generated_at_utc": wrapped.get("generated_at_utc"),
        "path": str(input_path),
    }


def save_runtime_state(app_state, output_path=PREPARED_RUNTIME_PATH):
    """Save only the expensive deterministic M02/M03/M04 runtime structures."""
    ensure_output_directories()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": app_state.get("corpus_sha256", ""),
        "dataset_audit": app_state.get("dataset_audit", {}),
        "processed_documents": app_state.get("processed_documents", []),
        "positioned_documents": app_state.get("positioned_documents", []),
        "index_data": app_state.get("index_data"),
        "build_errors": app_state.get("build_errors", []),
    }

    with gzip.open(output_path, "wb", compresslevel=5) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path


def load_runtime_state(input_path=PREPARED_RUNTIME_PATH):
    """Load a trusted runtime cache produced by save_runtime_state()."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Prepared runtime not found: {input_path}")

    with gzip.open(input_path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Prepared runtime payload is invalid.")

    return payload, {
        "schema_version": payload.get("schema_version"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "path": str(input_path),
    }


def save_manifest(manifest, output_path=PREPARED_MANIFEST_PATH):
    ensure_output_directories()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )

    return output_path


def load_manifest(input_path=PREPARED_MANIFEST_PATH):
    input_path = Path(input_path)
    if not input_path.exists():
        return {}
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
