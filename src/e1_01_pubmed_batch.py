import math
import time

from m01_input_parser import (
    extract_pmids_from_text,
    fetch_pubmed_documents,
)


# ============================================================
# EXTENSION E11 — PUBMED BATCH FETCH
# ============================================================
#
# Parent Module:
#     M01 — Input / Parser
#
# Purpose:
#     Robustly retrieve a corpus of hundreds to ~1,000 PubMed records
#     without putting every PMID into one EFetch URL.
#
# Design:
#     - Split PMID list into batches (default: 100)
#     - Preserve requested order
#     - Reuse M01's official PubMed EFetch + XML parser
#     - Retry transient failures
#     - Add pacing between batches
#     - Return a reproducibility report
# ============================================================


def normalize_pmids(pmids):
    requested = []
    seen = set()

    for value in pmids or []:
        pmid = str(value).strip()
        if not pmid:
            continue
        if not pmid.isdigit():
            raise ValueError(f"Invalid PMID: {value}")
        if pmid in seen:
            continue
        seen.add(pmid)
        requested.append(pmid)

    return requested


def iter_batches(values, batch_size=100):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def fetch_pubmed_documents_batched(
    pmids,
    batch_size=100,
    timeout=30,
    pause_seconds=0.35,
    max_retries=3,
    source_name="PubMed Batch Fetch",
):
    requested = normalize_pmids(pmids)

    if not requested:
        raise ValueError("No valid PMID was provided.")

    batch_size = int(batch_size)
    max_retries = max(1, int(max_retries))
    pause_seconds = max(0.0, float(pause_seconds))

    documents_by_pmid = {}
    missing_pmids = []
    batch_reports = []

    started = time.perf_counter()
    batches = list(iter_batches(requested, batch_size=batch_size))

    for batch_number, batch in enumerate(batches, start=1):
        batch_started = time.perf_counter()
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                documents, missing = fetch_pubmed_documents(
                    batch,
                    timeout=timeout,
                    source_name=(
                        f"{source_name} "
                        f"[batch {batch_number}/{len(batches)}]"
                    ),
                )

                for document in documents:
                    pmid = str(document.get("pmid", "")).strip()
                    if pmid:
                        documents_by_pmid[pmid] = document

                missing_pmids.extend(missing)

                batch_reports.append(
                    {
                        "batch_number": batch_number,
                        "requested": len(batch),
                        "returned": len(documents),
                        "missing": len(missing),
                        "attempts": attempt,
                        "elapsed_seconds": time.perf_counter() - batch_started,
                    }
                )

                last_error = None
                break

            except Exception as error:
                last_error = error

                if attempt < max_retries:
                    # Gentle exponential retry delay: 1s, 2s, 4s ...
                    time.sleep(min(8.0, 2 ** (attempt - 1)))

        if last_error is not None:
            raise RuntimeError(
                "PubMed batch fetch failed after "
                f"{max_retries} attempts for batch {batch_number}/"
                f"{len(batches)}: {last_error}"
            ) from last_error

        if batch_number < len(batches) and pause_seconds > 0:
            time.sleep(pause_seconds)

    # Preserve original requested order.
    ordered_documents = [
        documents_by_pmid[pmid]
        for pmid in requested
        if pmid in documents_by_pmid
    ]

    missing_set = set(missing_pmids)
    missing_ordered = [
        pmid
        for pmid in requested
        if pmid in missing_set or pmid not in documents_by_pmid
    ]

    elapsed = time.perf_counter() - started

    report = {
        "requested_count": len(requested),
        "returned_count": len(ordered_documents),
        "missing_count": len(missing_ordered),
        "batch_size": batch_size,
        "batch_count": len(batches),
        "pause_seconds": pause_seconds,
        "max_retries": max_retries,
        "elapsed_seconds": elapsed,
        "batches": batch_reports,
    }

    return ordered_documents, missing_ordered, report


def load_pubmed_documents_batched_from_text(
    text,
    batch_size=100,
    timeout=30,
    pause_seconds=0.35,
    max_retries=3,
    source_name="README.txt",
):
    pmids = extract_pmids_from_text(text)

    if not pmids:
        raise ValueError(
            "The text file does not contain a recognizable PMID."
        )

    return fetch_pubmed_documents_batched(
        pmids,
        batch_size=batch_size,
        timeout=timeout,
        pause_seconds=pause_seconds,
        max_retries=max_retries,
        source_name=source_name,
    )


if __name__ == "__main__":
    # Network calls are intentionally not executed automatically.
    demo = [str(10000000 + i) for i in range(250)]
    batches = list(iter_batches(demo, batch_size=100))
    assert [len(batch) for batch in batches] == [100, 100, 50]
    print("E11 batch partition test: PASS")
