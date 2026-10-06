import csv
import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from project2_config import (
    DEFAULT_GLP1_JATS_DIR,
    PREPARED_CORPUS_PATH,
    ensure_output_directories,
)


# ============================================================
# EXTENSION E12 — LOCAL JATS ABSTRACT CORPUS LOADER
# ============================================================
#
# Parent responsibility:
#     M01 / Input Adapter
#
# Purpose:
#     Read the fixed Project 2 GLP-1 JATS collection downloaded by:
#         PubMed search -> PMID -> PMC mapping -> PMC JATS XML
#
# IMPORTANT:
#     Project 2 analyzes scientific ABSTRACTS, not full-text bodies.
#     This loader therefore creates an M01-compatible document where
#     the search/statistics corpus is ONLY the selected main abstract.
#
# Main-abstract policy:
#     1. Prefer <abstract> without abstract-type.
#     2. Exclude precis/highlights/graphical/supplementary summaries.
#     3. If no untyped abstract exists, use the first non-excluded one.
#     4. Use paragraph body text; structured section headings are stored
#        separately but excluded from the term-frequency corpus.
#
# This policy is deterministic and is written into the audit metadata so
# local and Streamlit Cloud runs can use exactly the same corpus.
# ============================================================


EXCLUDED_ABSTRACT_TYPE_MARKERS = (
    "author-highlights",
    "highlight",
    "highlights",
    "precis",
    "graphical",
    "teaser",
    "plain-language",
    "plain language",
    "key-points",
    "key points",
    "publisher",
    "summary-for",
    "supplement",
)

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def _text(element):
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _article_id(article_meta, id_type):
    if article_meta is None:
        return ""

    wanted = str(id_type).casefold()
    for node in article_meta.findall("./{*}article-id"):
        if (node.attrib.get("pub-id-type") or "").casefold() == wanted:
            return _text(node)
    return ""


def _authors(article_meta):
    output = []
    if article_meta is None:
        return output

    for contrib in article_meta.findall(".//{*}contrib[@contrib-type='author']"):
        name = contrib.find("./{*}name")
        if name is None:
            collab = _text(contrib.find("./{*}collab"))
            if collab:
                output.append(collab)
            continue

        surname = _text(name.find("./{*}surname"))
        given = _text(name.find("./{*}given-names"))
        full = " ".join(part for part in (given, surname) if part).strip()
        if full:
            output.append(full)

    return output


def _affiliations(article_meta):
    if article_meta is None:
        return []

    seen = set()
    output = []
    for node in article_meta.findall(".//{*}aff"):
        value = _text(node)
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _keywords(article_meta):
    if article_meta is None:
        return []

    output = []
    seen = set()
    for node in article_meta.findall(".//{*}kwd-group/{*}kwd"):
        value = _text(node)
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _normalized_abstract_type(node):
    return (
        node.attrib.get("abstract-type")
        or node.attrib.get("content-type")
        or ""
    ).strip().casefold()


def _is_excluded_abstract(node):
    abstract_type = _normalized_abstract_type(node)
    return any(marker in abstract_type for marker in EXCLUDED_ABSTRACT_TYPE_MARKERS)


def _select_main_abstract(article_meta):
    if article_meta is None:
        return None, {
            "abstract_candidates": 0,
            "selected_abstract_type": "",
            "selection_reason": "no article-meta",
        }

    candidates = list(article_meta.findall("./{*}abstract"))
    usable = [node for node in candidates if not _is_excluded_abstract(node)]

    untyped = [node for node in usable if not _normalized_abstract_type(node)]

    if untyped:
        selected = untyped[0]
        reason = "preferred untyped main abstract"
    elif usable:
        selected = usable[0]
        reason = "fallback first non-excluded abstract"
    else:
        selected = None
        reason = "no non-excluded abstract"

    return selected, {
        "abstract_candidates": len(candidates),
        "usable_abstract_candidates": len(usable),
        "selected_abstract_type": (
            _normalized_abstract_type(selected)
            if selected is not None
            else ""
        ),
        "selection_reason": reason,
    }


def _extract_abstract_content(abstract_node):
    if abstract_node is None:
        return [], [], []

    section_titles = []
    sections = []
    paragraphs = []

    # Structured abstracts commonly use <sec><title>...<p>...</p></sec>.
    # Keep headings for display/audit but exclude headings from the term corpus.
    sec_nodes = list(abstract_node.findall("./{*}sec"))

    if sec_nodes:
        for sec in sec_nodes:
            label = _text(sec.find("./{*}title"))
            sec_paragraphs = [
                _text(p)
                for p in sec.findall(".//{*}p")
                if _text(p)
            ]

            if not sec_paragraphs:
                # Rare fallback: section text not wrapped in <p>.
                value = _text(sec)
                if label and value.startswith(label):
                    value = value[len(label):].strip()
                if value:
                    sec_paragraphs = [value]

            if label:
                section_titles.append(label)

            for value in sec_paragraphs:
                paragraphs.append(value)
                sections.append(
                    {
                        "label": label,
                        "raw_label": label,
                        "nlm_category": "",
                        "text": value,
                    }
                )
    else:
        direct_paragraphs = [
            _text(p)
            for p in abstract_node.findall(".//{*}p")
            if _text(p)
        ]

        if not direct_paragraphs:
            value = _text(abstract_node)
            if value:
                direct_paragraphs = [value]

        for value in direct_paragraphs:
            paragraphs.append(value)
            sections.append(
                {
                    "label": "",
                    "raw_label": "",
                    "nlm_category": "",
                    "text": value,
                }
            )

    return section_titles, paragraphs, sections


def parse_jats_abstract_only(xml_path):
    """Parse one PMC JATS file into the Project 2 abstract-only document contract."""
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    article = root
    if not str(article.tag).endswith("article"):
        article = root.find(".//{*}article")

    if article is None:
        raise ValueError("No <article> element found.")

    front = article.find("./{*}front")
    if front is None:
        raise ValueError("JATS article has no <front>.")

    article_meta = front.find("./{*}article-meta")
    if article_meta is None:
        raise ValueError("JATS article has no <article-meta>.")

    pmcid = _article_id(article_meta, "pmcid") or _article_id(article_meta, "pmc")
    pmid = _article_id(article_meta, "pmid")
    doi = _article_id(article_meta, "doi")

    title = _text(article_meta.find("./{*}title-group/{*}article-title"))
    journal = _text(
        front.find("./{*}journal-meta/{*}journal-title-group/{*}journal-title")
    )

    language = (
        article.attrib.get(XML_LANG)
        or article.attrib.get("xml:lang")
        or ""
    ).strip()

    abstract_node, selection = _select_main_abstract(article_meta)
    section_titles, paragraphs, sections = _extract_abstract_content(abstract_node)
    abstract_text = " ".join(paragraphs).strip()

    return {
        "filename": xml_path.name,
        "source_format": "JATS",
        "corpus_source": "PubMed-selected -> direct PMC JATS XML (pubmed_pmc)",
        "search_scope": "abstract_only",
        "analysis_scope": "main_abstract_body_only",
        "article_type": article.attrib.get("article-type", ""),
        "language": language,
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": doi,
        "journal": journal,
        "title": title,
        "authors": _authors(article_meta),
        "affiliations": _affiliations(article_meta),
        "keywords": _keywords(article_meta),
        "abstract_section_titles": section_titles,
        "abstract_sections": sections,
        "abstract_paragraphs": paragraphs,
        "abstract_paragraphs_clean": list(paragraphs),
        "abstract": abstract_text,
        "abstract_clean": abstract_text,
        # Full text intentionally excluded from the Project 2 corpus.
        "section_titles": [],
        "body_paragraphs": [],
        "body_paragraphs_clean": [],
        "body": "",
        "body_clean": "",
        "definitions": [],
        "figures": [],
        "tables": [],
        "acknowledgments": [],
        "references": [],
        # M02 Document Statistics is also abstract-only for this corpus.
        "statistics_front_text": "",
        "statistics_body_text": abstract_text,
        "statistics_back_text": "",
        "statistics_text": abstract_text,
        "statistics_scope": (
            "Project 2 main JATS abstract body text only "
            "(structured section headings excluded)"
        ),
        # JATS <word-count> generally refers to the article, not this abstract,
        # so it is deliberately not used for the Project 2 abstract corpus.
        "reported_word_count": None,
        "reported_figure_count": None,
        "reported_table_count": None,
        "reported_ref_count": None,
        "reported_page_count": None,
        "abstract_selection": selection,
    }


def _manifest_order(jats_dir):
    """Use the downloader manifest when available; otherwise deterministic filename order."""
    jats_dir = Path(jats_dir)
    candidates = [
        # Corrected V2 corpus manifests. The valid_1200 manifest is already
        # ordered by original PubMed relevance rank.
        jats_dir.parent / "reports" / f"{jats_dir.name}_manifest.csv",
        jats_dir.parent / "reports" / "valid_1200_manifest.csv",
        jats_dir.parent / "reports" / "formal_1000_manifest.csv",
        # Legacy downloader manifests.
        jats_dir.parent / "glp1_pubmed_jats_manifest.csv",
        jats_dir / "glp1_pubmed_jats_manifest.csv",
    ]

    manifest = next((path for path in candidates if path.exists()), None)
    if manifest is None:
        return None

    ordered = []
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                file_value = (row.get("file") or "").strip()
                pmcid = (row.get("pmcid") or "").strip()
                name = Path(file_value).name if file_value else (f"{pmcid}.xml" if pmcid else "")
                if name:
                    path = jats_dir / name
                    if path.exists():
                        ordered.append(path)
    except Exception:
        return None

    return ordered or None


def load_local_jats_corpus(
    jats_dir=DEFAULT_GLP1_JATS_DIR,
    target=1000,
    require_pmid=True,
    require_abstract=True,
    require_english=True,
):
    """
    Load and audit the official Project 2 GLP-1 local corpus.

    Returns:
        documents, audit
    """
    jats_dir = Path(jats_dir)
    if not jats_dir.exists():
        raise FileNotFoundError(f"JATS directory not found: {jats_dir}")

    xml_files = _manifest_order(jats_dir)
    if xml_files is None:
        xml_files = sorted(jats_dir.glob("*.xml"), key=lambda p: p.name.casefold())

    audit = {
        "source_directory": str(jats_dir),
        "target_documents": int(target) if target is not None else None,
        "xml_files_found": len(xml_files),
        "parsed_files": 0,
        "accepted_documents": 0,
        "missing_pmid": [],
        "missing_abstract": [],
        "non_english": [],
        "duplicate_pmids": [],
        "parse_errors": [],
        "excluded_abstract_type_counts": {},
        "abstract_policy": (
            "Prefer untyped main <abstract>; exclude precis/highlights/graphical/"
            "supplementary-style abstracts; analyze paragraph body text only."
        ),
        "analysis_scope": "main abstract body only; section headings excluded",
    }

    documents = []
    seen_pmids = set()

    for xml_path in xml_files:
        if target is not None and len(documents) >= int(target):
            break

        try:
            document = parse_jats_abstract_only(xml_path)
            audit["parsed_files"] += 1
        except Exception as error:
            audit["parse_errors"].append(
                {
                    "file": xml_path.name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        pmid = (document.get("pmid") or "").strip()
        abstract = (document.get("abstract") or "").strip()
        language = (document.get("language") or "").strip().casefold()

        if require_pmid and not pmid:
            audit["missing_pmid"].append(xml_path.name)
            continue

        if pmid and pmid in seen_pmids:
            audit["duplicate_pmids"].append(pmid)
            continue

        if require_abstract and not abstract:
            audit["missing_abstract"].append(xml_path.name)
            continue

        if require_english and language and not language.startswith("en"):
            audit["non_english"].append(
                {"file": xml_path.name, "language": language}
            )
            continue

        if pmid:
            seen_pmids.add(pmid)

        documents.append(document)

    audit["accepted_documents"] = len(documents)
    audit["unique_pmids"] = len(seen_pmids)
    audit["complete_target"] = (
        target is None or len(documents) >= int(target)
    )

    return documents, audit



# ============================================================
# GLP-1 REPRESENTATION AUDIT — EXACT EXPERIMENT CORPUS
# ============================================================
# Performed after the official Project 2 loader accepts the documents, so
# numerator and denominator always refer to the exact corpus used by all
# downstream experiments. This is a raw-main-abstract audit only; it does not
# modify preprocessing.
# ============================================================

GLP1_REPRESENTATION_PATTERNS = {
    "glp-1_ascii_exact": r"(?<![a-z0-9])glp\-1(?![a-z0-9])",
    "glp‐1_u2010_exact": r"(?<![a-z0-9])glp‐1(?![a-z0-9])",
    "glp‑1_u2011_exact": r"(?<![a-z0-9])glp‑1(?![a-z0-9])",
    "glp–1_u2013_exact": r"(?<![a-z0-9])glp–1(?![a-z0-9])",
    "glp_1_space_exact": r"(?<![a-z0-9])glp\s+1(?![a-z0-9])",
    "glp-1r_ascii_exact": r"(?<![a-z0-9])glp\-1r(?![a-z0-9])",
    "glucagon-like_peptide-1_ascii": r"(?<![a-z0-9])glucagon\-like\s+peptide\-1(?![a-z0-9])",
    "glucagon-like_peptide_1_ascii": r"(?<![a-z0-9])glucagon\-like\s+peptide\s+1(?![a-z0-9])",
    "glp−1_u2212_exact": r"(?<![a-z0-9])glp−1(?![a-z0-9])",
    "glp—1_u2014_exact": r"(?<![a-z0-9])glp—1(?![a-z0-9])",
    "glp‐1r_u2010_exact": r"(?<![a-z0-9])glp‐1r(?![a-z0-9])",
    "glp1_exact": r"(?<![a-z0-9])glp1(?![a-z0-9])",
    "glp1r_exact": r"(?<![a-z0-9])glp1r(?![a-z0-9])",
    "glucagon_like_peptide_1_spaces": r"(?<![a-z0-9])glucagon\s+like\s+peptide\s+1(?![a-z0-9])",
}


def audit_glp1_representations(documents):
    """Audit common GLP-1 forms in the exact system-accepted corpus."""
    documents = list(documents or [])
    compiled = {
        name: re.compile(pattern, re.IGNORECASE)
        for name, pattern in GLP1_REPRESENTATION_PATTERNS.items()
    }
    document_frequency = {name: 0 for name in compiled}
    any_form_ids = set()
    unmatched = []

    for index, document in enumerate(documents):
        abstract = str(document.get("abstract") or "")
        pmid = str(document.get("pmid") or "").strip()
        matched = False
        for name, pattern in compiled.items():
            if pattern.search(abstract):
                document_frequency[name] += 1
                matched = True
        if matched:
            any_form_ids.add(pmid or f"__row_{index}")
        else:
            unmatched.append({
                "pmid": pmid,
                "pmcid": str(document.get("pmcid") or ""),
                "title": str(document.get("title") or ""),
                "abstract_preview": abstract[:500],
            })

    total = len(documents)
    any_count = len(any_form_ids)
    return {
        "scope": "exact Project 2 accepted experiment corpus; raw main abstract only",
        "total_documents": total,
        "document_frequency": document_frequency,
        "any_form_unique_documents": any_count,
        "any_form_coverage": (any_count / total if total else 0.0),
        "documents_without_audited_form": len(unmatched),
        "unmatched_preview": unmatched[:10],
    }


def corpus_sha256(documents):
    """Stable hash over the exact PMID + main-abstract text used in analysis."""
    digest = hashlib.sha256()
    for document in documents or []:
        record = {
            "pmid": document.get("pmid", ""),
            "pmcid": document.get("pmcid", ""),
            "title": document.get("title", ""),
            "abstract": document.get("abstract", ""),
        }
        payload = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def save_prepared_corpus(documents, audit, output_path=PREPARED_CORPUS_PATH):
    """Save a portable, compressed corpus for identical local/cloud analysis."""
    ensure_output_directories()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "record_type": "project2_corpus_header",
        "schema_version": 1,
        "corpus_sha256": corpus_sha256(documents),
        "audit": audit,
    }

    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for document in documents or []:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")

    return output_path


def load_prepared_corpus(input_path=PREPARED_CORPUS_PATH):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Prepared corpus not found: {input_path}")

    with gzip.open(input_path, "rt", encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            raise ValueError("Prepared corpus is empty.")

        header = json.loads(first)
        documents = [json.loads(line) for line in handle if line.strip()]

    audit = header.get("audit", {})
    expected_hash = header.get("corpus_sha256", "")
    actual_hash = corpus_sha256(documents)

    audit = dict(audit)
    audit["prepared_corpus_path"] = str(input_path)
    audit["corpus_sha256"] = actual_hash
    audit["hash_verified"] = bool(expected_hash and expected_hash == actual_hash)
    audit["accepted_documents"] = len(documents)

    return documents, audit


if __name__ == "__main__":
    docs, report = load_local_jats_corpus()
    print("E12 - LOCAL JATS ABSTRACT CORPUS")
    print("Accepted:", len(docs))
    print("Audit:", json.dumps(report, ensure_ascii=False, indent=2))
    if docs:
        print("First PMID:", docs[0].get("pmid"))
        print("First title:", docs[0].get("title"))
        print("First abstract chars:", len(docs[0].get("abstract", "")))
