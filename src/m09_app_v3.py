import csv
import html
import io
import json
import math
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from wordcloud import WordCloud

from m02_text_processor import tokenize_words
from e2_01_porter import porter_stem

from m08_controller import (
    empty_app_state,
    build_from_local_jats,
    build_from_uploaded_files,
    load_prepared_project_state,
    prepared_project_available,
    get_project_paths,
    run_required_analysis_suite,
    run_preprocessing_analysis,
    run_preprocessing_zipf_analysis,
    run_term_statistics_analysis,
    run_all_term_statistics,
    run_document_tfidf_analysis,
    run_significant_words_analysis,
    run_zipf_analysis,
    run_word2vec_similar,
    run_spelling_correction,
    run_semantic_expansion,
    run_search,
    run_optional_domain_comparison_from_local_jats,
    get_index_summary,
    get_document_stats,
    get_extension_status,
)


# ============================================================
# MODULE 09 — STREAMLIT PROJECT 2 WORKBENCH
# ============================================================
#
# Design goals:
#     1. Prepared demo mode: identical local/cloud report numbers.
#     2. Rebuild mode: prove the full pipeline can recompute results.
#     3. Report workbench: every required number/table/plot is visible.
#     4. Legacy Project 2 features remain available:
#        Porter, Word2Vec, Edit Distance/Spelling, Semantic Expansion.
#     5. Optional two-domain comparison is supported without creating
#        a second IR system.
#
# M09 contains presentation logic only. All computations go through M08.
# ============================================================


st.set_page_config(
    page_title="NCKU Project 2 — RETRIEVAL",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# VISUAL SYSTEM — PROJECT 1 NATURAL GREEN, PRESENTATION V3
# ============================================================

st.markdown(
    r"""
    <style>
    :root {
        --p2-forest: #2E4031;
        --p2-forest-dark: #1C2321;
        --p2-moss: #8FBC8F;
        --p2-linen: #F4F6F0;
        --p2-soft: #E8EEE6;
        --p2-soft-2: #EEF2EC;
        --p2-card: #FFFFFF;
        --p2-ink: #1C2321;
        --p2-muted: #667269;
        --p2-border: #D7E0D4;
    }

    .stApp {
        background: var(--p2-linen);
        color: var(--p2-ink);
    }
    [data-testid="stHeader"] {
        background: rgba(244,246,240,0.95);
    }
    [data-testid="stSidebar"] {
        background: var(--p2-soft-2);
        border-right: 1px solid var(--p2-border);
    }

    .block-container {
        padding-top: 2.8rem;
        padding-bottom: 3rem;
    }

    .p2-header {
        margin: 0 0 1.05rem 0;
        padding: 0.70rem 0 0.9rem 0;
        border-bottom: 1px solid var(--p2-border);
    }
    .p2-kicker {
        color: var(--p2-forest);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        line-height: 1.45;
        padding-top: 0.15rem;
        margin-bottom: 0.25rem;
    }
    .p2-title {
        color: var(--p2-ink);
        font-size: clamp(2.5rem, 4vw, 4.25rem);
        line-height: 0.98;
        font-weight: 850;
        letter-spacing: -0.035em;
        margin: 0;
    }
    .p2-subtitle {
        color: var(--p2-forest);
        font-size: 1.08rem;
        font-weight: 750;
        margin-top: 0.55rem;
    }
    .p2-scope {
        color: var(--p2-muted);
        font-size: 0.86rem;
        margin-top: 0.45rem;
    }


    /* Project 2 RETRIEVAL hero — visual continuation of Project 1 MATCH. */
    .retrieval-hero {
        margin: 0 0 1.0rem 0;
        padding: 1.25rem 1.35rem 1.15rem 1.35rem;
        border: 1px solid rgba(143,188,143,0.28);
        border-radius: 13px;
        background: linear-gradient(105deg, #1C2921 0%, #263C2D 54%, #45694A 100%);
        box-shadow: 0 14px 36px rgba(0,0,0,0.20);
    }
    .retrieval-kicker {
        color: #D7E3D4;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.32rem;
    }
    .retrieval-title {
        color: #FFFFFF;
        /* Match Project 1 hero title scale exactly. */
        font-size: clamp(2rem, 3vw, 3rem);
        line-height: 1.08;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 13px 0;
    }
    .retrieval-subtitle {
        color: #D6E2D4;
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 0.55rem;
    }
    .retrieval-flow {
        display: flex;
        flex-wrap: wrap;
        gap: 0.38rem;
        margin-top: 0.75rem;
    }
    .retrieval-flow span {
        display: inline-flex;
        align-items: center;
        min-height: 1.7rem;
        padding: 0.20rem 0.58rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.13);
        color: #EEF4EC;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .retrieval-search-heading {
        font-size: 1.55rem;
        line-height: 1.1;
        font-weight: 850;
        letter-spacing: -0.02em;
        margin: 1.25rem 0 0.20rem 0;
    }
    .retrieval-search-subtitle {
        color: var(--p2-muted);
        font-size: 0.86rem;
        margin: 0 0 0.75rem 0;
    }
    .retrieval-mini-label {
        color: var(--p2-forest);
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin: 0.55rem 0 0.30rem 0;
    }

    /* Folder-tab look for the main report navigation. */
    [data-baseweb="tab-list"] {
        gap: 0.2rem;
        border-bottom: 2px solid var(--p2-forest);
        align-items: flex-end;
    }
    [data-baseweb="tab"] {
        background: var(--p2-soft);
        border: 1px solid var(--p2-border);
        border-bottom: none;
        border-radius: 10px 10px 0 0;
        padding: 0.48rem 0.78rem !important;
        min-height: 2.55rem;
        color: var(--p2-forest-dark);
        font-weight: 720;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: var(--p2-forest);
        color: #FFFFFF !important;
        border-color: var(--p2-forest);
        transform: translateY(1px);
    }
    [data-baseweb="tab"][aria-selected="true"] * {
        color: #FFFFFF !important;
    }
    [data-baseweb="tab-highlight"] {
        background: transparent !important;
    }

    /* Stateful navigation uses segmented controls so reruns keep the selected page. */
    [data-testid="stSegmentedControl"] {
        margin: 0.15rem 0 0.75rem 0;
    }
    [data-testid="stSegmentedControl"] button {
        border-color: var(--p2-border) !important;
        background: var(--p2-soft) !important;
        color: var(--p2-forest-dark) !important;
        font-weight: 720 !important;
        min-height: 2.4rem;
    }
    [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background: var(--p2-forest) !important;
        color: #FFFFFF !important;
        border-color: var(--p2-forest) !important;
    }
    [data-testid="stSegmentedControl"] button[aria-pressed="true"] * {
        color: #FFFFFF !important;
    }

    .p2-section-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1.1rem 0 0.9rem 0;
    }
    .p2-section-no {
        background: var(--p2-forest-dark);
        color: #FFFFFF;
        border-radius: 9px;
        min-width: 42px;
        height: 42px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        font-weight: 850;
    }
    .p2-section-name {
        font-size: 1.7rem;
        font-weight: 850;
        color: var(--p2-ink);
        line-height: 1.05;
    }
    .p2-section-note {
        color: var(--p2-muted);
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }

    .p2-table-wrap {
        overflow-x: auto;
        background: var(--p2-card);
        border: 1px solid var(--p2-border);
        border-radius: 14px;
        padding: 0.25rem;
        box-shadow: 0 4px 14px rgba(46,64,49,0.05);
    }
    table.p2-table {
        width: 100%;
        border-collapse: collapse;
        color: var(--p2-ink);
        font-size: 0.92rem;
    }
    table.p2-table th {
        background: var(--p2-soft);
        color: var(--p2-forest-dark);
        text-align: left;
        padding: 0.68rem 0.65rem;
        border-bottom: 1px solid var(--p2-border);
        font-weight: 800;
        white-space: nowrap;
    }
    table.p2-table td {
        padding: 0.62rem 0.65rem;
        border-bottom: 1px solid #E7ECE5;
        vertical-align: middle;
    }
    table.p2-table tr:last-child td {
        border-bottom: none;
    }
    .p2-table-wrap.p2-compact {
        width: fit-content;
        min-width: 900px;
        max-width: 1120px;
    }
    .p2-table-wrap.p2-compact table.p2-table {
        width: auto;
        min-width: 900px;
    }
    .p2-table-wrap.p2-compact table.p2-table th,
    .p2-table-wrap.p2-compact table.p2-table td {
        padding-left: 0.85rem;
        padding-right: 0.85rem;
    }
    .p2-table-wrap.p2-compact table.p2-table th:first-child,
    .p2-table-wrap.p2-compact table.p2-table td:first-child {
        min-width: 220px;
    }

    .p2-term-card {
        background: var(--p2-card);
        border: 1px solid var(--p2-border);
        border-radius: 14px;
        padding: 0.9rem 1rem 0.75rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 5px 16px rgba(46,64,49,0.06);
    }
    .p2-term-card-title {
        color: var(--p2-forest);
        font-size: 1.05rem;
        font-weight: 850;
        margin-bottom: 0.5rem;
    }
    table.p2-term-table {
        width: 100%;
        border-collapse: collapse;
    }
    table.p2-term-table th {
        color: var(--p2-muted);
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        text-align: left;
        padding: 0.35rem 0.35rem;
        border-bottom: 1px solid var(--p2-border);
    }
    table.p2-term-table td {
        padding: 0.36rem 0.35rem;
        border-bottom: 1px solid #EDF0EB;
    }
    .p2-term-rank {
        width: 2.8rem;
        color: var(--p2-muted);
        font-weight: 800;
    }
    .p2-term-word {
        color: var(--p2-ink);
        font-weight: 800;
        line-height: 1.0;
    }
    .p2-term-num {
        color: var(--p2-muted);
        font-variant-numeric: tabular-nums;
    }
    .p2-top3 {
        color: #B42318 !important;
        font-weight: 900 !important;
    }

    /* Page 02 Part III — fixed Top-50 comparison geometry.
       Keep every A/B/C/D table aligned across rank-range changes and
       keep CF/DF visually close to the term column for presentation. */
    .p2-cf50-card {
        min-height: 0;
    }
    .p2-cf50-table-wrap {
        width: 78%;
        max-width: 78%;
        min-width: 0;
        overflow-x: visible;
    }
    table.p2-cf50-table {
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
    }
    table.p2-cf50-table th {
        color: var(--p2-muted);
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        text-align: left;
        padding: 0.35rem 0.35rem;
        border-bottom: 1px solid var(--p2-border);
    }
    table.p2-cf50-table td {
        padding: 0.36rem 0.35rem;
        border-bottom: 1px solid #EDF0EB;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    table.p2-cf50-table col.p2-cf50-rank { width: 9%; }
    table.p2-cf50-table col.p2-cf50-term { width: 43%; }
    table.p2-cf50-table col.p2-cf50-cf   { width: 24%; }
    table.p2-cf50-table col.p2-cf50-df   { width: 24%; }
    table.p2-cf50-table .p2-term-rank {
        width: auto;
    }
    @media (max-width: 1100px) {
        .p2-cf50-table-wrap {
            width: 100%;
            max-width: 100%;
        }
    }

    .p2-callout {
        background: var(--p2-soft);
        border-left: 4px solid var(--p2-forest);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.6rem 0 1rem 0;
        color: var(--p2-ink);
    }

    .stButton > button, .stFormSubmitButton > button {
        border-radius: 9px !important;
        font-weight: 750 !important;
    }

    /* Stable project-wide loading indicator: simple circular spinner. */
    .p2-loading {
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
        color: var(--p2-muted);
        font-size: 0.9rem;
        padding: 0.35rem 0;
    }
    .p2-spinner {
        width: 18px;
        height: 18px;
        border: 2.5px solid var(--p2-border);
        border-top-color: var(--p2-forest);
        border-radius: 50%;
        display: inline-block;
        box-sizing: border-box;
        animation: p2-spin 0.8s linear infinite;
        flex: 0 0 auto;
    }
    @keyframes p2-spin {
        to { transform: rotate(360deg); }
    }

    /* Streamlit's native script-running status appears beside Deploy.
       Replace its animated figure/icon with the same neutral circular spinner. */
    [data-testid="stStatusWidget"] {
        position: relative !important;
        width: 24px !important;
        min-width: 24px !important;
        height: 24px !important;
        min-height: 24px !important;
        padding: 0 !important;
        margin: 0 0.35rem !important;
        overflow: visible !important;
    }
    [data-testid="stStatusWidget"] > * {
        visibility: hidden !important;
    }
    [data-testid="stStatusWidget"]::after {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 16px;
        height: 16px;
        margin: -8px 0 0 -8px;
        border: 2px solid var(--p2-border);
        border-top-color: var(--p2-forest);
        border-radius: 50%;
        box-sizing: border-box;
        animation: p2-spin 0.8s linear infinite;
        pointer-events: none;
    }

    @media (max-width: 900px) {
        [data-baseweb="tab"] {
            padding: 0.38rem 0.55rem !important;
            font-size: 0.78rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 1. SESSION STATE
# ============================================================


def _init_state():
    defaults = {
        "app_state": empty_app_state(),
        "auto_load_attempted": False,
        "timings": {},
        "search_result": None,
        "spell_search_result": None,
        "tfidf_result": None,
        "w2v_similar_result": None,
        "w2v_similarity_result": None,
        "home_search_result": None,
        "semantic_search_result": None,
        "porter_mini_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def _time_call(label, function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    st.session_state["timings"][label] = time.perf_counter() - started
    return result


def _format_seconds(value):
    if value is None:
        return "—"
    value = float(value)
    if value < 1:
        return f"{value * 1000:.0f} ms"
    return f"{value:.2f} s"


def _fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _csv_bytes(rows):
    rows = list(rows or [])
    if not rows:
        return b""
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def _safe_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


@contextmanager
def _circular_spinner(message):
    """Project-styled circular loading indicator without changing task logic."""
    placeholder = st.empty()
    placeholder.markdown(
        (
            '<div class="p2-loading" role="status" aria-live="polite">'
            '<span class="p2-spinner" aria-hidden="true"></span>'
            f'<span>{html.escape(str(message))}</span>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()


# ============================================================
# 2. AUTO-LOAD PREPARED PROJECT
# ============================================================


if (
    not st.session_state["auto_load_attempted"]
    and prepared_project_available()
):
    st.session_state["auto_load_attempted"] = True
    try:
        with _circular_spinner("Loading prepared GLP-1 Project 2 corpus and report results..."):
            st.session_state["app_state"] = _time_call(
                "load_prepared_project",
                load_prepared_project_state,
            )
    except Exception as error:
        st.session_state["app_state"] = empty_app_state()
        st.session_state["app_state"]["build_errors"].append(
            {
                "filename": "prepared project",
                "error": f"{type(error).__name__}: {error}",
            }
        )


app_state = st.session_state["app_state"]
paths = get_project_paths()


def _ensure_v3_log10_statistics(state):
    """Keep the presentation consistent even if an older prepared artifact is loaded.

    Recomputes only CF/DF/IDF derived tables from the already-loaded A-D
    preprocessing result. It does not rebuild the corpus, index, Zipf regression,
    or Word2Vec model.
    """
    preprocessing = state.get("preprocessing_result") or {}
    stats_by_condition = state.get("term_statistics_by_condition") or {}
    if not preprocessing:
        return

    needs_refresh = (
        not stats_by_condition
        or any((result or {}).get("log_base") != 10 for result in stats_by_condition.values())
    )
    if not needs_refresh:
        return

    try:
        run_all_term_statistics(
            state,
            cf_df_top_n=50,
            idf_top_n=20,
            log_base=10,
        )
        # Significant Words uses IDF; refresh it after changing the log base.
        run_significant_words_analysis(state, top_n=20)
    except Exception as error:
        state.setdefault("build_errors", []).append(
            {
                "filename": "V3 log10 refresh",
                "error": f"{type(error).__name__}: {error}",
            }
        )


_ensure_v3_log10_statistics(app_state)


# ============================================================
# 3. PLOTTING / PRESENTATION HELPERS
# ============================================================


def _zipf_rows(result):
    return list((result or {}).get("zipf", []) or [])


def _plot_rank_frequency(result, title="Rank–Frequency Distribution", chart_key="zipf_rank_frequency", height=470):
    rows = _zipf_rows(result)
    if not rows:
        st.info("No Zipf rows available.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=[row["rank"] for row in rows],
            y=[row["frequency"] for row in rows],
            mode="lines",
            name="Observed CF",
            hovertemplate="Rank=%{x}<br>CF=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Rank",
        yaxis_title="Collection Frequency (CF)",
        height=height,
    )
    # A log y-axis keeps the long tail visible while this remains a direct
    # rank-vs-frequency plot. The next figure is the explicit log-log experiment.
    fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch", key=chart_key)


def _plot_loglog_with_regression(
    result,
    title="Log–Log Zipf Plot + Linear Regression",
    chart_key="zipf_loglog_regression",
    height=500,
    show_high_region_fit=False,
):
    """Plot observed log-log points and the frozen regression fits.

    ``show_high_region_fit`` is presentation-only.  When enabled, the chart
    overlays the already prepared High-frequency regional regression from
    ``rank_region_analysis``.  No regression is recomputed here: the stored
    slope / intercept / raw-rank boundaries are used directly.
    """
    rows = _zipf_rows(result)
    regression = (result or {}).get("regression", {}) or {}
    if not rows:
        st.info("No Zipf rows available.")
        return

    x = [row["log_rank"] for row in rows]
    y = [row["log_frequency"] for row in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker={"size": 4, "opacity": 0.55},
            name="Observed",
            text=[row["term"] for row in rows],
            customdata=[row["rank"] for row in rows],
            hovertemplate=(
                "Term=%{text}<br>Rank=%{customdata}<br>"
                "log10(rank)=%{x:.4f}<br>log10(CF)=%{y:.4f}<extra></extra>"
            ),
        )
    )

    if regression.get("valid"):
        slope = regression.get("slope")
        intercept = regression.get("intercept")
        x_min, x_max = min(x), max(x)
        fit_x = [x_min, x_max]
        fit_y = [intercept + slope * value for value in fit_x]
        fig.add_trace(
            go.Scatter(
                x=fit_x,
                y=fit_y,
                mode="lines",
                name="Overall fit" if show_high_region_fit else "Least-squares fit",
                line={"width": 2.5, "dash": "solid"},
                hovertemplate="Overall fit<br>log10(rank)=%{x:.4f}<br>log10(CF)=%{y:.4f}<extra></extra>",
            )
        )

    if show_high_region_fit:
        regions = ((result or {}).get("rank_region_analysis", {}) or {}).get("regions", {}) or {}
        high = regions.get("high") or {}
        high_slope = high.get("slope")
        high_intercept = high.get("intercept")
        rank_start = high.get("rank_start")
        rank_end = high.get("rank_end")

        # The High region is defined in raw-rank space by the prepared analysis.
        # Only its already-computed regression line is drawn in log-log space.
        if (
            high.get("valid")
            and high_slope is not None
            and high_intercept is not None
            and rank_start is not None
            and rank_end is not None
            and float(rank_start) > 0
            and float(rank_end) > 0
        ):
            high_x = [math.log10(float(rank_start)), math.log10(float(rank_end))]
            high_y = [
                float(high_intercept) + float(high_slope) * value
                for value in high_x
            ]
            fig.add_trace(
                go.Scatter(
                    x=high_x,
                    y=high_y,
                    mode="lines",
                    name="High-region fit",
                    line={"width": 3.0, "dash": "dash"},
                    hovertemplate=(
                        "High-region fit<br>"
                        "log10(rank)=%{x:.4f}<br>log10(CF)=%{y:.4f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="log10(rank)",
        yaxis_title="log10(CF)",
        height=height,
        margin=dict(l=55, r=25, t=78, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
        ),
    )
    st.plotly_chart(fig, width="stretch", key=chart_key)


def _plot_top_terms(rows, title="Top Terms", chart_key="top_terms"):
    rows = list(rows or [])
    if not rows:
        st.info("No term rows available.")
        return

    terms = [row.get("term", "") for row in rows][::-1]
    values = [
        row.get("collection_frequency", row.get("frequency", 0))
        for row in rows
    ][::-1]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=terms,
            orientation="h",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Collection Frequency",
        yaxis_title="Term",
        height=max(420, 22 * len(rows) + 120),
    )
    st.plotly_chart(fig, width="stretch", key=chart_key)


def _plot_preprocessing_comparison(rows, key_prefix="preprocessing_comparison"):
    rows = list(rows or [])
    if not rows:
        return

    labels = [row.get("condition", "") for row in rows]

    fig1 = go.Figure()
    fig1.add_trace(
        go.Bar(
            x=labels,
            y=[row.get("vocabulary_size", 0) for row in rows],
            name="Vocabulary",
        )
    )
    fig1.update_layout(
        title="Vocabulary Size by Preprocessing Condition",
        xaxis_title="Condition",
        yaxis_title="Unique Terms",
        height=420,
    )
    st.plotly_chart(fig1, width="stretch", key=f"{key_prefix}_vocabulary")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=labels,
            y=[row.get("zipf_exponent") for row in rows],
            mode="lines+markers",
            name="Zipf exponent",
        )
    )
    fig2.update_layout(
        title="Estimated Zipf Exponent by Condition",
        xaxis_title="Condition",
        yaxis_title="Zipf exponent k",
        height=420,
    )
    st.plotly_chart(fig2, width="stretch", key=f"{key_prefix}_exponent")


def _plot_zipf_exponent_comparison(rows, chart_key="zipf_exponent_comparison"):
    """Presentation chart for Part VI: show the A → B → C → D trend in Zipf k.

    Uses the already prepared/frozen comparison rows only; no regression values
    are recalculated here.
    """
    rows = list(rows or [])
    if not rows:
        return

    labels = [_condition_short_label(row.get("condition", "")) for row in rows]
    values = [row.get("zipf_exponent") for row in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers+text",
            text=[_fmt(value, 4) for value in values],
            textposition="top center",
            line=dict(width=3),
            marker=dict(size=10),
            name="Zipf exponent k",
            hovertemplate="%{x}<br>k=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Estimated Zipf Exponent by Condition",
        xaxis_title="Condition",
        yaxis_title="Zipf exponent k",
        height=420,
        margin=dict(l=55, r=30, t=80, b=65),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", key=chart_key)


def _plot_cf_df(rows, chart_key="cf_df_scatter"):
    rows = list(rows or [])
    if not rows:
        return

    fig = go.Figure(
        go.Scattergl(
            x=[row.get("document_frequency", 0) for row in rows],
            y=[row.get("collection_frequency", 0) for row in rows],
            mode="markers",
            text=[row.get("term", "") for row in rows],
            customdata=[row.get("idf") for row in rows],
            hovertemplate=(
                "Term=%{text}<br>DF=%{x}<br>CF=%{y}<br>IDF=%{customdata:.4f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Collection Frequency vs Document Frequency",
        xaxis_title="Document Frequency (DF)",
        yaxis_title="Collection Frequency (CF)",
        height=480,
    )
    if all(row.get("document_frequency", 0) > 0 for row in rows):
        fig.update_xaxes(type="log")
    if all(row.get("collection_frequency", 0) > 0 for row in rows):
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch", key=chart_key)


def _plot_porter_comparison(result, chart_key="porter_comparison"):
    result = result or {}
    original = result.get("original", {})
    stemmed = result.get("stemmed", {})
    original_rows = _zipf_rows(original)
    stemmed_rows = _zipf_rows(stemmed)
    if not original_rows or not stemmed_rows:
        st.info("Porter + Zipf comparison is not ready.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=[row["log_rank"] for row in original_rows],
            y=[row["log_frequency"] for row in original_rows],
            mode="lines",
            name="Original",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=[row["log_rank"] for row in stemmed_rows],
            y=[row["log_frequency"] for row in stemmed_rows],
            mode="lines",
            name="Porter stemmed",
        )
    )
    fig.update_layout(
        title="Before vs After Porter — Log–Log Distribution",
        xaxis_title="log10(rank)",
        yaxis_title="log10(CF)",
        height=480,
    )
    st.plotly_chart(fig, width="stretch", key=chart_key)



def _normalize_word2vec_key(word):
    return (
        str(word or "")
        .casefold()
        .replace("’", "'")
        .strip()
    )


def _word2vec_visual_rows(similar, max_terms=15):
    """Return a bounded list of valid Word2Vec neighbor rows for presentation charts."""
    rows = []
    for row in list(similar or []):
        term = row.get("word", row.get("term", ""))
        similarity = row.get("similarity")
        if not term or similarity is None:
            continue
        try:
            score = float(similarity)
        except (TypeError, ValueError):
            continue
        rows.append({"word": str(term), "similarity": score})
        if len(rows) >= int(max_terms):
            break
    return rows


def _plot_word2vec_similarity_bars(similar, query_word, chart_key="word2vec_similarity_bars"):
    rows = _word2vec_visual_rows(similar, max_terms=20)
    if not rows:
        st.info("No similar-word scores are available for the chart.")
        return

    labels = [row["word"] for row in rows][::-1]
    scores = [row["similarity"] for row in rows][::-1]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            text=[f"{value:.3f}" for value in scores],
            textposition="auto",
            hovertemplate="Term=%{y}<br>Cosine similarity=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Most Similar Words to '{query_word}'",
        xaxis_title="Cosine Similarity (original Word2Vec space)",
        yaxis_title="Term",
        height=max(430, 30 * len(rows) + 150),
        margin=dict(l=30, r=20, t=70, b=60),
    )
    st.plotly_chart(fig, width="stretch", key=chart_key)


def _pca_word2vec_projection(model, query_word, similar, max_terms=12):
    """
    Project query + nearest Word2Vec neighbors from the original embedding
    space into 2D using a small deterministic PCA/SVD calculation.

    No sklearn dependency is required. Similarity ranking remains the original
    Word2Vec cosine similarity; PCA is presentation-only.
    """
    try:
        import numpy as np
    except Exception:
        return []

    query_key = _normalize_word2vec_key(query_word)
    if not query_key or query_key not in model.wv:
        return []

    neighbors = _word2vec_visual_rows(similar, max_terms=max_terms)
    words = [query_key]
    similarity_by_word = {query_key: 1.0}

    for row in neighbors:
        candidate = _normalize_word2vec_key(row["word"])
        if not candidate or candidate == query_key or candidate not in model.wv:
            continue
        words.append(candidate)
        similarity_by_word[candidate] = float(row["similarity"])

    # PCA needs at least three points to make a useful 2D presentation.
    if len(words) < 3:
        return []

    matrix = np.vstack([np.asarray(model.wv[word], dtype=float) for word in words])
    centered = matrix - matrix.mean(axis=0, keepdims=True)

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return []

    if vh.shape[0] < 2:
        return []

    components = vh[:2].copy()

    # Resolve the arbitrary PCA sign so the same input is visually stable.
    for component_index in range(2):
        component = components[component_index]
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            components[component_index] *= -1

    coords = centered @ components.T

    return [
        {
            "word": word,
            "pc1": float(coords[i, 0]),
            "pc2": float(coords[i, 1]),
            "similarity": float(similarity_by_word.get(word, 0.0)),
            "is_query": i == 0,
        }
        for i, word in enumerate(words)
    ]


def _plot_word2vec_pca(model, query_word, similar, chart_key="word2vec_pca"):
    rows = _pca_word2vec_projection(
        model=model,
        query_word=query_word,
        similar=similar,
        max_terms=12,
    )

    if not rows:
        st.info("PCA visualization needs the query word and at least two valid neighbors in the Word2Vec vocabulary.")
        return

    query_rows = [row for row in rows if row["is_query"]]
    neighbor_rows = [row for row in rows if not row["is_query"]]

    fig = go.Figure()

    if neighbor_rows:
        fig.add_trace(
            go.Scatter(
                x=[row["pc1"] for row in neighbor_rows],
                y=[row["pc2"] for row in neighbor_rows],
                mode="markers+text",
                text=[row["word"] for row in neighbor_rows],
                textposition="top center",
                name="Similar words",
                customdata=[[row["word"], row["similarity"]] for row in neighbor_rows],
                hovertemplate=(
                    "Term=%{customdata[0]}<br>"
                    "Cosine similarity=%{customdata[1]:.4f}<br>"
                    "PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>"
                ),
            )
        )

    if query_rows:
        row = query_rows[0]
        fig.add_trace(
            go.Scatter(
                x=[row["pc1"]],
                y=[row["pc2"]],
                mode="markers+text",
                text=[row["word"]],
                textposition="bottom center",
                marker=dict(symbol="star", size=18),
                name="Query word",
                hovertemplate=(
                    "Query=%{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="2D PCA Projection of Word2Vec Neighbors",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        height=560,
        legend_title_text="Embedding points",
        margin=dict(l=40, r=20, t=70, b=60),
    )
    st.plotly_chart(fig, width="stretch", key=chart_key)


def _wordcloud(frequency):
    frequency = {
        str(term): int(count)
        for term, count in (frequency or {}).items()
        if int(count) > 0
    }
    if not frequency:
        st.info("No frequency data available for word cloud.")
        return

    cloud = WordCloud(
        width=1500,
        height=700,
        background_color="white",
        collocations=False,
        max_words=150,
    ).generate_from_frequencies(frequency)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def _safe_highlight(context, matched_text):
    context = context or ""
    matched_text = matched_text or ""
    if not matched_text:
        return html.escape(context)
    lower_context = context.casefold()
    lower_match = matched_text.casefold()
    start = lower_context.find(lower_match)
    if start < 0:
        return html.escape(context)
    end = start + len(matched_text)
    return (
        html.escape(context[:start])
        + "<mark>"
        + html.escape(context[start:end])
        + "</mark>"
        + html.escape(context[end:])
    )


def _document_metadata_lookup(state):
    """Map PMCID / PMID / filename identifiers to the loaded document metadata."""
    lookup = {}
    for document in (state or {}).get("positioned_documents", []) or []:
        for key in (document.get("pmcid"), document.get("pmid"), document.get("filename")):
            if key not in (None, ""):
                lookup[str(key)] = document
    return lookup


def _render_search_result(
    result,
    heading="Search Results",
    state=None,
    result_key="search",
    page_size=10,
):
    """Render a paginated search result without building every result widget at once.

    Retrieval still computes the complete result set.  Pagination only limits the
    Streamlit UI to the current page so common terms (for example GLP-1) do not
    create hundreds of expanders and thousands of Markdown elements on every rerun.
    """
    if result is None:
        st.info("No search has been executed yet.")
        return

    st.markdown(f"### {heading}")
    st.caption(f"Query: {result.get('query', '')}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieved Documents", result.get("documents_found", 0))
    c2.metric("Exact Matches", result.get("exact_matches", 0))
    c3.metric("Related Matches", result.get("related_matches", 0))
    c4.metric("Total Matches", result.get("total_matches", 0))

    document_results = list(result.get("document_results", []) or [])
    if not document_results:
        st.info("No matching documents were returned.")
        return

    page_size = max(1, int(page_size or 10))
    total_documents = len(document_results)
    total_pages = max(1, math.ceil(total_documents / page_size))
    page_key = f"{result_key}_page"

    current_page = int(st.session_state.get(page_key, 1) or 1)
    current_page = max(1, min(current_page, total_pages))
    st.session_state[page_key] = current_page

    start_index = (current_page - 1) * page_size
    end_index = min(start_index + page_size, total_documents)

    nav_left, nav_center, nav_right = st.columns([1, 3, 1])
    with nav_left:
        if st.button(
            "← Previous",
            key=f"{result_key}_prev",
            disabled=(current_page <= 1),
            width="stretch",
        ):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with nav_center:
        st.markdown(
            f"<div style='text-align:center;padding:.45rem 0;'>"
            f"<b>Page {current_page} / {total_pages}</b> · "
            f"Showing documents {start_index + 1}–{end_index} of {total_documents}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with nav_right:
        if st.button(
            "Next →",
            key=f"{result_key}_next",
            disabled=(current_page >= total_pages),
            width="stretch",
        ):
            st.session_state[page_key] = current_page + 1
            st.rerun()

    metadata_lookup = _document_metadata_lookup(state) if state else {}
    page_results = document_results[start_index:end_index]

    for page_offset, document_result in enumerate(page_results):
        rank = document_result.get("document_rank", start_index + page_offset + 1)
        title = document_result.get("title", "") or "Untitled"
        matches = document_result.get("total_matches", 0)
        document_id = document_result.get("document_id", "")
        metadata = metadata_lookup.get(str(document_id), {})
        pmid = metadata.get("pmid", "")

        # Only ten documents are mounted in the Streamlit component tree at once.
        # The first result on the current page is opened for immediate presentation.
        with st.expander(
            f"{rank}. {title} — {matches} matches",
            expanded=(page_offset == 0),
        ):
            identity_parts = []
            if pmid:
                identity_parts.append(f"PMID: {pmid}")
            identity_parts.append(f"Document ID: {document_id}")
            identity_parts.append(f"Exact: {document_result.get('exact_matches', 0)}")
            identity_parts.append(f"Related: {document_result.get('related_matches', 0)}")
            st.caption(" · ".join(identity_parts))

            for item in document_result.get("results", []):
                context_html = _safe_highlight(
                    item.get("context", ""),
                    item.get("matched_text", ""),
                )
                matched_text = item.get("matched_text", "")
                st.markdown(
                    (
                        f"**{item.get('match_method', '').replace('_', ' ').title()}** · "
                        f"Match `{matched_text}` · "
                        f"Field `{item.get('field', '')}` · "
                        f"Word `{item.get('word_position')}` · "
                        f"Sentence `{item.get('sentence_position')}` · "
                        f"Char `{item.get('char_start')}:{item.get('char_end')}`"
                    )
                )
                st.markdown(
                    f'<div style="padding:.55rem .75rem;border-left:3px solid #78927d;">{context_html}</div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# 4. SIDEBAR — DATA / REBUILD CONTROLS
# ============================================================


with st.sidebar:
    st.header("Corpus / Demo Mode")

    if paths["prepared_corpus_exists"]:
        st.success("Prepared corpus available")
        if st.button("Load Prepared Project", width="stretch"):
            try:
                with _circular_spinner("Loading prepared project..."):
                    st.session_state["app_state"] = _time_call(
                        "load_prepared_project",
                        load_prepared_project_state,
                    )
                    app_state = st.session_state["app_state"]
                st.rerun()
            except Exception as error:
                st.error(f"Load failed: {type(error).__name__}: {error}")
    else:
        st.warning("Prepared corpus not found")
        st.code("python src\\project2_prepare_all.py", language=None)

    if paths["raw_glp1_jats_exists"]:
        if st.button("Rebuild from Downloaded JATS", width="stretch"):
            try:
                with _circular_spinner("Reading JATS and rebuilding M02 → M04..."):
                    st.session_state["app_state"] = _time_call(
                        "rebuild_raw_jats",
                        build_from_local_jats,
                    )
                    app_state = st.session_state["app_state"]
                    st.session_state["search_result"] = None
                st.rerun()
            except Exception as error:
                st.error(f"Rebuild failed: {type(error).__name__}: {error}")

    st.divider()
    st.markdown("**Ad-hoc demo input**")
    uploads = st.file_uploader(
        "PMID TXT or XML",
        type=["txt", "xml"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Build Uploaded Files", width="stretch"):
        try:
            with _circular_spinner("Building uploaded documents..."):
                st.session_state["app_state"] = _time_call(
                    "build_uploaded",
                    build_from_uploaded_files,
                    uploads,
                )
                app_state = st.session_state["app_state"]
            st.rerun()
        except Exception as error:
            st.error(f"Build failed: {type(error).__name__}: {error}")

    st.divider()
    if app_state.get("index_ready"):
        if st.button("Run All Required Analyses", type="primary", width="stretch"):
            try:
                with _circular_spinner("Running A-D, Zipf, CF/DF/IDF and Porter..."):
                    _time_call(
                        "required_analysis_suite",
                        run_required_analysis_suite,
                        app_state,
                    )
                st.rerun()
            except Exception as error:
                st.error(f"Analysis failed: {type(error).__name__}: {error}")

    with st.expander("Paths", expanded=False):
        st.json(paths)


app_state = st.session_state["app_state"]

if not app_state.get("index_ready"):
    st.info(
        "No corpus is loaded. Run `python src\\project2_prepare_all.py` once, "
        "or rebuild from the local JATS folder."
    )
    if app_state.get("build_errors"):
        st.json(app_state.get("build_errors"))
    st.stop()


summary = get_index_summary(app_state)
audit = app_state.get("dataset_audit", {}) or {}
ext_status = get_extension_status(app_state)



# ============================================================
# 5. PRESENTATION HELPERS — V2
# ============================================================


def _question_block(title, questions, note=None):
    st.markdown(f"### {title}")
    for item in questions:
        st.markdown(f"- {item}")
    if note:
        st.caption(note)


def _condition_default_index(keys, preferred="B_no_punctuation"):
    keys = list(keys or [])
    try:
        return keys.index(preferred)
    except ValueError:
        return 0


def _condition_label(preprocessing, key):
    return (
        (preprocessing or {})
        .get("conditions", {})
        .get(key, {})
        .get("label", key)
    )


def _clean_comparison_rows(rows):
    cleaned = []
    for row in rows or []:
        cleaned.append(
            {
                "Condition": row.get("condition"),
                "Description": row.get("label"),
                "Documents": row.get("documents"),
                "Tokens": row.get("total_tokens"),
                "Vocabulary": row.get("vocabulary_size"),
                "Zipf k": row.get("zipf_exponent"),
                "Slope": row.get("slope"),
                "R²": row.get("r_squared"),
                "RMSE": row.get("rmse"),
                "Best region": row.get("best_rank_region"),
            }
        )
    return cleaned


def _get_report_zipf(app_state, preferred="B_no_punctuation"):
    prep_zipf = app_state.get("preprocessing_zipf_result") or {}
    by_condition = prep_zipf.get("conditions", {}) or {}
    if preferred in by_condition:
        return by_condition[preferred], preferred
    if app_state.get("zipf_result"):
        return app_state.get("zipf_result"), "M04 baseline"
    return None, preferred



def _section_header(number, title, note=""):
    note_html = (
        f'<div class="p2-section-note">{html.escape(note)}</div>'
        if note
        else ""
    )
    st.markdown(
        f"""
        <div class="p2-section-title">
          <div class="p2-section-no">{html.escape(str(number))}</div>
          <div>
            <div class="p2-section-name">{html.escape(title)}</div>
            {note_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _html_table(headers, rows, align_right=None, compact=False):
    align_right = set(align_right or [])
    wrap_class = "p2-table-wrap p2-compact" if compact else "p2-table-wrap"
    parts = [f'<div class="{wrap_class}"><table class="p2-table"><thead><tr>']
    for idx, header in enumerate(headers):
        style = ' style="text-align:right"' if idx in align_right else ''
        parts.append(f'<th{style}>{html.escape(str(header))}</th>')
    parts.append('</tr></thead><tbody>')
    for row in rows:
        parts.append('<tr>')
        for idx, value in enumerate(row):
            style = ' style="text-align:right"' if idx in align_right else ''
            parts.append(f'<td{style}>{html.escape(str(value))}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


def _condition_short_label(key):
    mapping = {
        "A_basic": "A Basic",
        "B_no_punctuation": "B Remove punctuation",
        "C_no_stopwords": "C Remove stopwords",
        "D_porter_stemming": "D Porter stemming",
    }
    return mapping.get(key, key)


def _build_ad_summary_rows(app_state):
    preprocessing = app_state.get("preprocessing_result") or {}
    prep_zipf = app_state.get("preprocessing_zipf_result") or {}
    zipf_by_condition = prep_zipf.get("conditions", {}) or {}
    rows = []
    for key in preprocessing.get("condition_order", []):
        condition = preprocessing.get("conditions", {}).get(key, {})
        regression = (zipf_by_condition.get(key) or {}).get("regression", {}) or {}
        rows.append(
            [
                _condition_short_label(key),
                f"{condition.get('total_tokens', 0):,}",
                f"{condition.get('unique_terms', 0):,}",
                f"{condition.get('average_tokens_per_document', 0.0):.3f}",
                _fmt(regression.get("zipf_exponent"), 4),
                _fmt(regression.get("r_squared"), 4),
                _fmt(regression.get("rmse"), 4),
            ]
        )
    return rows


def _render_ad_summary_table(app_state):
    rows = _build_ad_summary_rows(app_state)
    if not rows:
        st.info("A–D preprocessing results are not ready.")
        return
    _html_table(
        ["Condition", "Tokens", "Vocabulary", "Avg tokens/doc", "Zipf k", "R²", "RMSE"],
        rows,
        align_right={1, 2, 3, 4, 5, 6},
        compact=True,
    )


def _render_preprocessing_summary_charts(app_state):
    """Page 02 presentation-only A–D token/vocabulary summaries.

    Values come directly from the frozen preprocessing_result; this helper does
    not recompute or modify any preprocessing metric.
    """
    preprocessing = app_state.get("preprocessing_result") or {}
    conditions = preprocessing.get("conditions", {}) or {}
    keys = [
        "A_basic",
        "B_no_punctuation",
        "C_no_stopwords",
        "D_porter_stemming",
    ]

    available = [key for key in keys if key in conditions]
    if not available:
        st.info("A–D preprocessing results are not ready.")
        return

    labels = [_condition_short_label(key) for key in available]
    tokens = [int((conditions.get(key) or {}).get("total_tokens", 0) or 0) for key in available]
    vocabulary = [int((conditions.get(key) or {}).get("unique_terms", 0) or 0) for key in available]

    left, right = st.columns(2)

    with left:
        token_fig = go.Figure(
            go.Bar(
                x=labels,
                y=tokens,
                text=[f"{value:,}" for value in tokens],
                textposition="outside",
                textfont=dict(size=15),
                cliponaxis=False,
                marker_color="#2E4031",
                hovertemplate="%{x}<br>Tokens=%{y:,}<extra></extra>",
            )
        )
        token_fig.update_layout(
            title="Tokens by Condition",
            height=340,
            margin=dict(l=45, r=20, t=55, b=85),
            showlegend=False,
            yaxis_title="Tokens",
            xaxis_title=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        token_fig.update_xaxes(tickangle=0, tickfont=dict(size=15))
        token_fig.update_yaxes(tickfont=dict(size=13))
        st.plotly_chart(token_fig, width="stretch", key="v3_page02_tokens_chart")

    with right:
        vocab_fig = go.Figure(
            go.Bar(
                x=labels,
                y=vocabulary,
                text=[f"{value:,}" for value in vocabulary],
                textposition="outside",
                textfont=dict(size=15),
                cliponaxis=False,
                marker_color="#2E4031",
                hovertemplate="%{x}<br>Vocabulary=%{y:,}<extra></extra>",
            )
        )
        vocab_fig.update_layout(
            title="Vocabulary by Condition",
            height=340,
            margin=dict(l=45, r=20, t=55, b=85),
            showlegend=False,
            yaxis_title="Unique Terms",
            xaxis_title=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        vocab_fig.update_xaxes(tickangle=0, tickfont=dict(size=15))
        vocab_fig.update_yaxes(tickfont=dict(size=13))
        st.plotly_chart(vocab_fig, width="stretch", key="v3_page02_vocabulary_chart")


def _render_zipf_regression_parameters(app_state):
    """Show every overall regression parameter required by the detailed spec."""
    comparison = (app_state.get("preprocessing_zipf_result") or {}).get("comparison", [])
    rows = []
    for row in comparison or []:
        rows.append(
            [
                _condition_short_label(row.get("condition", "")),
                _fmt(row.get("slope"), 4),
                _fmt(row.get("intercept"), 4),
                _fmt(row.get("zipf_exponent"), 4),
                _fmt(row.get("r_squared"), 4),
                _fmt(row.get("rmse"), 4),
            ]
        )
    if not rows:
        st.info("Overall regression parameters are not ready.")
        return
    _html_table(
        ["Condition", "Slope", "Intercept", "Zipf k", "R²", "RMSE"],
        rows,
        align_right={1, 2, 3, 4, 5},
        compact=True,
    )
    st.caption("Overall regression: log10(CF) = Intercept + Slope × log10(Rank); Zipf k = −Slope.")


def _term_font_size(rank):
    # V3 rehearsal revision: keep every term the same size for easier
    # side-by-side reading. Rank 1-3 are emphasized by color instead.
    return 16


def _term_card_html(title, rows, include_idf=False, include_score=False):
    headers = ["Rank", "Term", "CF", "DF"]
    if include_idf:
        headers.append("IDF")
    if include_score:
        headers.append("Score")

    out = [
        '<div class="p2-term-card">',
        f'<div class="p2-term-card-title">{html.escape(title)}</div>',
        '<table class="p2-term-table"><thead><tr>',
    ]
    for h in headers:
        out.append(f'<th>{html.escape(h)}</th>')
    out.append('</tr></thead><tbody>')

    for row in rows:
        rank = int(row.get("rank", 0) or 0)
        term = row.get("term", "")
        cf = int(row.get("collection_frequency", 0) or 0)
        df = int(row.get("document_frequency", 0) or 0)
        size = _term_font_size(rank)
        out.append('<tr>')
        top_class = " p2-top3" if rank in (1, 2, 3) else ""
        out.append(f'<td class="p2-term-rank{top_class}">{rank}</td>')
        out.append(
            f'<td><span class="p2-term-word{top_class}" style="font-size:{size}px">'
            f'{html.escape(str(term))}</span></td>'
        )
        out.append(f'<td class="p2-term-num">{cf:,}</td>')
        out.append(f'<td class="p2-term-num">{df:,}</td>')
        if include_idf:
            out.append(f'<td class="p2-term-num">{_fmt(row.get("idf"), 4)}</td>')
        if include_score:
            out.append(
                f'<td class="p2-term-num">{_fmt(row.get("candidate_resolving_score"), 4)}</td>'
            )
        out.append('</tr>')

    out.append('</tbody></table></div>')
    return ''.join(out)


def _render_top_terms_grid(app_state, top_n=10):
    preprocessing = app_state.get("preprocessing_result") or {}
    keys = list(preprocessing.get("condition_order", []))
    if not keys:
        st.info("Preprocessing results are not ready.")
        return

    for row_start in range(0, len(keys), 2):
        cols = st.columns(2)
        for offset, key in enumerate(keys[row_start:row_start + 2]):
            condition = preprocessing.get("conditions", {}).get(key, {})
            rows = list(condition.get("top_terms", []) or [])[:top_n]
            with cols[offset]:
                st.markdown(
                    _term_card_html(_condition_short_label(key), rows),
                    unsafe_allow_html=True,
                )


def _render_term_across_conditions(app_state, term):
    term = str(term or "").casefold().strip()
    if not term:
        return
    stats_by_condition = app_state.get("term_statistics_by_condition", {}) or {}
    rows = []
    for key in ["A_basic", "B_no_punctuation", "C_no_stopwords", "D_porter_stemming"]:
        stats = stats_by_condition.get(key) or {}
        match = next(
            (row for row in stats.get("rows", []) if row.get("term") == term),
            None,
        )
        if match:
            rows.append(
                [
                    _condition_short_label(key),
                    f"{match.get('collection_frequency', 0):,}",
                    f"{match.get('document_frequency', 0):,}",
                    f"{match.get('document_coverage', 0.0):.1%}",
                    _fmt(match.get("idf"), 5),
                ]
            )
        else:
            rows.append([_condition_short_label(key), "—", "—", "—", "—"])
    _html_table(
        ["Condition", "CF", "DF", "Coverage", "IDF log10"],
        rows,
        align_right={1,2,3,4},
    )


def _cf_top50_ranked_rows(app_state, condition_key):
    """Return deterministic CF-descending Rank 1–50 rows for one condition.

    Page 02 Part III uses the frozen term-level CF / DF values only.  Rank is
    reconstructed independently inside each preprocessing condition so the UI
    never depends on a missing/placeholder rank field in cf_df_rows.
    """
    stats_by_condition = app_state.get("term_statistics_by_condition", {}) or {}
    stats = stats_by_condition.get(condition_key) or {}
    source_rows = list(stats.get("cf_df_rows", []) or [])

    ranked = sorted(
        source_rows,
        key=lambda row: (
            -int(row.get("collection_frequency", 0) or 0),
            str(row.get("term", "")).casefold(),
        ),
    )[:50]

    output = []
    for rank, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = rank
        output.append(item)
    return output


def _cf_top50_card_html(title, rows):
    """Page 02 Part III card with fixed column geometry.

    This is intentionally separate from the generic term-card renderer so
    Other report pages and the interactive demo keep their existing presentation.
    """
    out = [
        '<div class="p2-term-card p2-cf50-card">',
        f'<div class="p2-term-card-title">{html.escape(title)}</div>',
        '<div class="p2-cf50-table-wrap">',
        '<table class="p2-cf50-table">',
        '<colgroup>',
        '<col class="p2-cf50-rank">',
        '<col class="p2-cf50-term">',
        '<col class="p2-cf50-cf">',
        '<col class="p2-cf50-df">',
        '</colgroup>',
        '<thead><tr><th>Rank</th><th>Term</th><th>CF</th><th>DF</th></tr></thead><tbody>',
    ]

    for row in rows:
        rank = int(row.get("rank", 0) or 0)
        term = str(row.get("term", ""))
        cf = int(row.get("collection_frequency", 0) or 0)
        df = int(row.get("document_frequency", 0) or 0)
        top_class = " p2-top3" if rank in (1, 2, 3) else ""
        out.append('<tr>')
        out.append(f'<td class="p2-term-rank{top_class}">{rank}</td>')
        out.append(
            f'<td title="{html.escape(term, quote=True)}">'
            f'<span class="p2-term-word{top_class}" style="font-size:16px">'
            f'{html.escape(term)}</span></td>'
        )
        out.append(f'<td class="p2-term-num">{cf:,}</td>')
        out.append(f'<td class="p2-term-num">{df:,}</td>')
        out.append('</tr>')

    out.append('</tbody></table></div></div>')
    return ''.join(out)


def _render_cf_top50_four_conditions(app_state, rank_start=1, rank_end=10):
    """Render the same selected Top-50 rank window for A / B / C / D."""
    keys = ["A_basic", "B_no_punctuation", "C_no_stopwords", "D_porter_stemming"]

    for row_start in range(0, 4, 2):
        cols = st.columns(2)
        for offset, key in enumerate(keys[row_start:row_start + 2]):
            ranked_rows = _cf_top50_ranked_rows(app_state, key)
            visible_rows = [
                row
                for row in ranked_rows
                if rank_start <= int(row.get("rank", 0) or 0) <= rank_end
            ]
            with cols[offset]:
                if visible_rows:
                    st.markdown(
                        _cf_top50_card_html(_condition_short_label(key), visible_rows),
                        unsafe_allow_html=True,
                    )
                else:
                    st.info(f"{_condition_short_label(key)} Top 50 CF data are not ready.")


def _render_rank_frequency_four_condition_grids(app_state):
    """Page 03 / Part V Experiment 1 — actual Rank vs Frequency plots."""
    prep_zipf = app_state.get("preprocessing_zipf_result") or {}
    by_condition = prep_zipf.get("conditions", {}) or {}
    keys = ["A_basic", "B_no_punctuation", "C_no_stopwords", "D_porter_stemming"]

    st.markdown("### Rank vs Frequency — A / B / C / D")
    for row_start in range(0, 4, 2):
        cols = st.columns(2)
        for offset, key in enumerate(keys[row_start:row_start + 2]):
            result = by_condition.get(key) or {}
            with cols[offset]:
                _plot_rank_frequency(
                    result,
                    title=f"{_condition_short_label(key)} — Rank vs Frequency",
                    chart_key=f"v3_zipf_rank_{key}",
                    height=360,
                )


def _render_loglog_four_condition_grids(app_state):
    """Page 03 / Part V Experiment 2 — observed log-log points + regression."""
    prep_zipf = app_state.get("preprocessing_zipf_result") or {}
    by_condition = prep_zipf.get("conditions", {}) or {}
    keys = ["A_basic", "B_no_punctuation", "C_no_stopwords", "D_porter_stemming"]

    st.markdown("### Log–Log Regression — A / B / C / D")
    for row_start in range(0, 4, 2):
        cols = st.columns(2)
        for offset, key in enumerate(keys[row_start:row_start + 2]):
            result = by_condition.get(key) or {}
            with cols[offset]:
                _plot_loglog_with_regression(
                    result,
                    title=f"{_condition_short_label(key)} — Log–Log + Regression",
                    chart_key=f"v3_zipf_loglog_{key}",
                    height=390,
                    show_high_region_fit=True,
                )


def _render_regional_r2_grouped_bar(app_state):
    """Presentation summary for Required Analysis Q4.

    Only High and Middle are plotted.  Low is intentionally excluded because
    the frozen long-tail region can have near-zero variance in log10(CF), so R²
    is undefined / not informative rather than a meaningful numeric zero.
    """
    prep_zipf = app_state.get("preprocessing_zipf_result") or {}
    by_condition = prep_zipf.get("conditions", {}) or {}
    keys = ["A_basic", "B_no_punctuation", "C_no_stopwords", "D_porter_stemming"]

    labels = []
    high_values = []
    middle_values = []

    for key in keys:
        result = by_condition.get(key) or {}
        regions = (result.get("rank_region_analysis", {}) or {}).get("regions", {}) or {}
        high = regions.get("high") or {}
        middle = regions.get("middle") or {}

        high_r2 = high.get("r_squared")
        middle_r2 = middle.get("r_squared")
        if high_r2 is None or middle_r2 is None:
            continue

        labels.append(_condition_short_label(key))
        high_values.append(float(high_r2))
        middle_values.append(float(middle_r2))

    if not labels:
        st.info("Regional R² summary is not ready.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="High-frequency region",
            x=labels,
            y=high_values,
            text=[f"{value:.4f}" for value in high_values],
            textposition="outside",
            textfont=dict(size=15),
            cliponaxis=False,
            hovertemplate="%{x}<br>High-frequency R²=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Middle-frequency region",
            x=labels,
            y=middle_values,
            text=[f"{value:.4f}" for value in middle_values],
            textposition="outside",
            textfont=dict(size=15),
            cliponaxis=False,
            hovertemplate="%{x}<br>Middle-frequency R²=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Regional Fit Comparison — R² Across A–D",
        barmode="group",
        height=430,
        margin=dict(l=45, r=25, t=70, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        xaxis=dict(
            title=dict(text="Preprocessing Condition", font=dict(size=15)),
            tickfont=dict(size=15),
        ),
        yaxis=dict(
            title=dict(text="R²", font=dict(size=15)),
            range=[0, 1.06],
            tickformat=".2f",
            tickfont=dict(size=13),
            gridcolor="rgba(120,130,120,0.15)",
        ),
    )
    st.plotly_chart(
        fig,
        width="stretch",
        key="v3_page04_regional_r2_grouped",
    )


def _build_region_comparison_rows(app_state):
    prep_zipf = app_state.get("preprocessing_zipf_result") or {}
    by_condition = prep_zipf.get("conditions", {}) or {}
    rows = []
    for key in ["A_basic", "B_no_punctuation", "C_no_stopwords", "D_porter_stemming"]:
        result = by_condition.get(key) or {}
        regions = (result.get("rank_region_analysis", {}) or {}).get("regions", {}) or {}
        for region_name in ["high", "middle", "low"]:
            value = regions.get(region_name) or {}
            if not value:
                continue
            rows.append([
                _condition_short_label(key),
                region_name.title(),
                value.get("rank_start", "—"),
                value.get("rank_end", "—"),
                _fmt(value.get("zipf_exponent"), 4),
                _fmt(value.get("r_squared"), 4),
                _fmt(value.get("rmse"), 4),
            ])
    return rows


def _render_significant_grid(app_state, top_n=10):
    results = app_state.get("significant_words_by_condition", {}) or {}
    if not results:
        st.info("Significant Words candidate analysis is not ready.")
        return
    keys = ["A_basic", "B_no_punctuation", "C_no_stopwords", "D_porter_stemming"]
    for row_start in range(0, 4, 2):
        cols = st.columns(2)
        for offset, key in enumerate(keys[row_start:row_start + 2]):
            result = results.get(key) or {}
            rows = list(result.get("candidate_terms", []) or [])[:top_n]
            with cols[offset]:
                st.markdown(
                    _term_card_html(
                        _condition_short_label(key),
                        rows,
                        include_idf=True,
                        include_score=True,
                    ),
                    unsafe_allow_html=True,
                )


def _conceptual_resolving_power_values(ranks, cutoff_range=None):
    """Return a smooth full-range conceptual resolving-power hump.

    The curve is dynamically parameterized by the illustrative Upper / Lower
    cut-offs, but the cut-offs are NOT the support boundaries of the curve.
    Resolving power remains defined across the entire rank range.  The peak
    follows the midpoint of the current cut-offs, while the right side decays
    more gradually to retain the lecture-style long-tail shape.

    This is a conceptual visualization only; it is not measured from corpus
    data and it does not alter frozen CF / DF / IDF or Zipf results.
    """
    safe_ranks = [max(1, int(rank or 1)) for rank in (ranks or [])]
    if not safe_ranks:
        return []

    max_rank = max(safe_ranks)
    if max_rank <= 1:
        return [1.0 for _ in safe_ranks]

    if cutoff_range:
        upper_rank, lower_rank = [int(value) for value in cutoff_range]
        if upper_rank > lower_rank:
            upper_rank, lower_rank = lower_rank, upper_rank
    else:
        # Presentation fallback only when illustrative cut-offs are hidden.
        upper_rank = max(1, int(round(max_rank * 0.05)))
        lower_rank = max(upper_rank + 1, int(round(max_rank * 0.25)))

    upper_rank = max(1, min(upper_rank, max_rank - 1))
    lower_rank = max(upper_rank + 1, min(lower_rank, max_rank))

    # Peak follows the midpoint of the user-controlled significant-word window.
    center = (float(upper_rank) + float(lower_rank)) / 2.0
    half_span = max(1.0, (float(lower_rank) - float(upper_rank)) / 2.0)

    # Smooth asymmetric generalized-Cauchy hump.
    # The left side rises from a low but non-zero value at Rank 1.
    # The right side falls more slowly and continues through the long tail.
    left_scale = max(1.0, half_span / 1.35)
    right_scale = max(1.0, half_span / 0.90)
    left_shape = 2.20
    right_shape = 1.80

    values = []
    for rank in safe_ranks:
        rank_value = float(rank)
        if rank_value <= center:
            distance = (center - rank_value) / left_scale
            value = 1.0 / (1.0 + distance ** left_shape)
        else:
            distance = (rank_value - center) / right_scale
            value = 1.0 / (1.0 + distance ** right_shape)
        values.append(value)

    # Normalize only for display on the secondary 0–1 axis.
    peak = max(values) if values else 1.0
    if peak <= 0:
        return [0.0 for _ in values]
    return [float(value) / float(peak) for value in values]


def _plot_significant_frequency_and_concept(
    app_state,
    condition_key,
    show_cutoffs=False,
    cutoff_range=None,
):
    """Render measured rank-frequency data plus a dynamic conceptual hump.

    The Collection Frequency trace is frozen corpus evidence.  The resolving-power
    hump is a lecture/Luhn-van-Rijsbergen conceptual visualization dynamically
    parameterized by the illustrative cut-offs; it is not a measured corpus metric.
    """
    results = app_state.get("significant_words_by_condition", {}) or {}
    result = results.get(condition_key) or {}
    rows = list(result.get("rows_by_rank", []) or [])
    if not rows:
        st.info("No Significant Words rows are available for this condition.")
        return

    ranks = [int(row.get("rank", 0) or 0) for row in rows]
    terms = [str(row.get("term", "")) for row in rows]
    cf = [int(row.get("collection_frequency", 0) or 0) for row in rows]
    df = [int(row.get("document_frequency", 0) or 0) for row in rows]
    idf = [float(row.get("idf", 0.0) or 0.0) for row in rows]

    active_cutoffs = cutoff_range if (show_cutoffs and cutoff_range) else None
    conceptual = _conceptual_resolving_power_values(ranks, active_cutoffs)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scattergl(
            x=ranks,
            y=cf,
            mode="lines",
            name="Actual Collection Frequency (CF)",
            text=terms,
            customdata=list(zip(df, idf)),
            hovertemplate=(
                "Term=%{text}<br>Rank=%{x}<br>CF=%{y:,}<br>"
                "DF=%{customdata[0]:,}<br>IDF=%{customdata[1]:.4f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ranks,
            y=conceptual,
            mode="lines",
            name="Conceptual Resolving-Power Curve (illustrative)",
            line={"width": 3, "dash": "dash"},
            hovertemplate=(
                "Rank=%{x}<br>Conceptual resolving power=%{y:.3f}"
                "<br><i>Illustrative only — not corpus measurement</i><extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    if active_cutoffs:
        upper_rank, lower_rank = [int(value) for value in active_cutoffs]
        if upper_rank > lower_rank:
            upper_rank, lower_rank = lower_rank, upper_rank

        # Shade only; labels are added separately so they never collide with the legend.
        fig.add_vrect(
            x0=upper_rank,
            x1=lower_rank,
            opacity=0.10,
            line_width=0,
            layer="below",
        )
        fig.add_vline(x=upper_rank, line_dash="dash", line_width=2)
        fig.add_vline(x=lower_rank, line_dash="dash", line_width=2)

        # Layer 3: cut-off labels inside the plot area.
        fig.add_annotation(
            x=upper_rank,
            y=0.92,
            xref="x",
            yref="paper",
            text="Illustrative Upper Cut-off",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font={"size": 11},
            bgcolor="rgba(255,255,255,0.72)",
        )
        fig.add_annotation(
            x=lower_rank,
            y=0.92,
            xref="x",
            yref="paper",
            text="Illustrative Lower Cut-off",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font={"size": 11},
            bgcolor="rgba(255,255,255,0.72)",
        )
        fig.add_annotation(
            x=(upper_rank + lower_rank) / 2.0,
            y=0.82,
            xref="x",
            yref="paper",
            text="Illustrative Significant-Words Region",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font={"size": 11},
            bgcolor="rgba(255,255,255,0.72)",
        )

    # Chart title is rendered as a Streamlit heading above Plotly so the title,
    # horizontal legend and plot annotations occupy three distinct vertical layers.
    st.markdown(
        f"#### {_condition_short_label(condition_key)} — Word Frequency vs. Conceptual Resolving Power"
    )

    fig.update_layout(
        xaxis_title="Words by Rank Order / Rank",
        height=650,
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.06,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.65)",
        },
        margin={"l": 65, "r": 65, "t": 125, "b": 55},
    )
    fig.update_yaxes(
        title_text="Actual Collection Frequency (CF)",
        type="log",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Conceptual Resolving Power (normalized 0–1)",
        range=[0.0, 1.08],
        secondary_y=True,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        key=f"v3_significant_frequency_concept_{condition_key}",
    )


def _candidate_significant_rows(
    app_state,
    condition_key,
    cutoff_range=None,
    top_n=50,
):
    """Return real corpus term statistics, filtered by the current rank window.

    Page 05 intentionally does not use or display the old exploratory proxy score.
    Rows are ordered by Rank so the table directly reflects the selected region of
    the actual rank-frequency distribution.
    """
    results = app_state.get("significant_words_by_condition", {}) or {}
    result = results.get(condition_key) or {}
    rows = list(result.get("rows_by_rank", []) or [])

    if cutoff_range:
        upper_rank, lower_rank = [int(value) for value in cutoff_range]
        if upper_rank > lower_rank:
            upper_rank, lower_rank = lower_rank, upper_rank
        rows = [
            row
            for row in rows
            if upper_rank <= int(row.get("rank", 0) or 0) <= lower_rank
        ]

    rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("rank", 10**18) or 10**18),
            str(row.get("term", "")),
        ),
    )[:max(0, int(top_n))]

    return [
        {
            "Rank": int(row.get("rank", 0) or 0),
            "Term": row.get("term", ""),
            "CF": int(row.get("collection_frequency", 0) or 0),
            "DF": int(row.get("document_frequency", 0) or 0),
            "IDF (log10)": round(float(row.get("idf", 0.0) or 0.0), 4),
        }
        for row in rows
    ]




# ============================================================
# PAGE 01 — RETRIEVAL HOME
# ============================================================


def _spell_correction_rows(spell):
    rows = []
    for item in (spell or {}).get("corrections", []) or []:
        corrected = item.get("corrected", "")
        distance = None
        for candidate in item.get("candidates", []) or []:
            if candidate.get("term") == corrected:
                distance = candidate.get("distance")
                break
        if distance is None and item.get("known"):
            distance = 0
        rows.append(
            {
                "Original": item.get("original", ""),
                "Suggested Correction": corrected,
                "Edit Distance": distance if distance is not None else "—",
            }
        )
    return rows


def _render_porter_mini_result(result):
    if not result:
        st.info("Enter text and click Stem to demonstrate the functional Porter module.")
        return

    st.markdown("### Porter Stemming")
    st.dataframe(result.get("rows", []), width="stretch", hide_index=True)

    preprocessing = app_state.get("preprocessing_result") or {}
    conditions = preprocessing.get("conditions", {}) or {}
    condition_c = conditions.get("C_no_stopwords") or {}
    condition_d = conditions.get("D_porter_stemming") or {}
    c_vocab = condition_c.get("unique_terms")
    d_vocab = condition_d.get("unique_terms")
    if c_vocab is not None and d_vocab is not None:
        st.caption(
            f"Formal pipeline check: Condition C vocabulary {int(c_vocab):,} → "
            f"Condition D (C + Porter) {int(d_vocab):,}."
        )


def _render_word2vec_home_result():
    similar = st.session_state.get("w2v_similar_result")
    query_word = st.session_state.get("w2v_similar_query", "")
    if similar is None:
        st.info("Enter a term and click Similar to inspect semantic neighbors in the prepared Word2Vec model.")
        return

    st.markdown("### Similar Words + Cosine Similarity")
    st.dataframe(similar, width="stretch", hide_index=True)

    if not similar:
        st.warning("No similar words were returned. The term may be outside the prepared Word2Vec vocabulary.")
        return

    st.markdown("### Word2Vec Vector Visualization")
    st.caption(
        "Similarity is calculated in the original Word2Vec vector space. "
        "The PCA plot is a 2D visualization only."
    )
    viz_left, viz_right = st.columns(2)
    with viz_left:
        _plot_word2vec_similarity_bars(
            similar=similar,
            query_word=query_word,
            chart_key="v3_home_w2v_similarity_bar",
        )
    with viz_right:
        _plot_word2vec_pca(
            model=app_state["word2vec_result"]["model"],
            query_word=query_word,
            similar=similar,
            chart_key="v3_home_w2v_pca",
        )


def _render_retrieval_home():
    st.markdown(
        '<div class="retrieval-search-heading">Search 1,000 GLP-1 Scientific Abstracts</div>'
        '<div class="retrieval-search-subtitle">Enter a biomedical term or query, choose a retrieval mode, and run it directly against the prepared corpus.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="retrieval-mini-label">Mode</div>', unsafe_allow_html=True)
    mode = _stateful_horizontal_selector(
        "Retrieval mode",
        RETRIEVAL_MODES,
        key="active_retrieval_mode",
    )

    option_values = {}
    if mode == "Spelling":
        with st.expander("Spelling options", expanded=False):
            a, b = st.columns(2)
            option_values["max_distance"] = a.number_input(
                "Max edit distance", 1, 5, 2, key="home_spell_distance"
            )
            option_values["candidate_topn"] = b.number_input(
                "Candidate terms", 1, 20, 5, key="home_spell_topn"
            )
    elif mode == "Word2Vec":
        option_values["topn"] = st.number_input(
            "Similar terms", 1, 30, 10, key="home_w2v_topn"
        )
    elif mode == "Semantic Expansion":
        with st.expander("Expansion options", expanded=False):
            a, b = st.columns(2)
            option_values["topn_per_word"] = a.number_input(
                "Neighbors per word", 1, 20, 3, key="home_semantic_topn"
            )
            option_values["min_similarity"] = b.slider(
                "Minimum cosine similarity", 0.0, 1.0, 0.5, 0.05, key="home_semantic_min"
            )

    button_labels = {
        "Keyword Search": "Search",
        "Spelling": "Correct",
        "Word2Vec": "Similar",
        "Semantic Expansion": "Expand",
        "Porter": "Stem",
    }
    placeholders = {
        "Keyword Search": "Enter a biomedical term or query...",
        "Spelling": "Try: gpp-1 or diabets",
        "Word2Vec": "Try: glp-1, diabetes, obesity, insulin",
        "Semantic Expansion": "Try: glp-1 or obesity treatment",
        "Porter": "Try: studies treatment receptors",
    }

    with st.form("retrieval_home_form", clear_on_submit=False):
        query_col, action_col = st.columns([8.7, 1.3])
        query = query_col.text_input(
            "Query",
            placeholder=placeholders[mode],
            key="retrieval_home_query",
            label_visibility="collapsed",
        )
        submitted = action_col.form_submit_button(
            button_labels[mode],
            type="primary",
            width="stretch",
        )

    if submitted:
        if not query.strip():
            st.warning("Please enter a query or text first.")
        elif mode == "Keyword Search":
            try:
                with _circular_spinner("Searching prepared corpus..."):
                    st.session_state["home_search_result"] = _time_call(
                        "home_search",
                        run_search,
                        app_state,
                        query,
                        "auto",
                    )
                    st.session_state["home_search_page"] = 1
            except Exception as error:
                st.error(f"Search failed: {type(error).__name__}: {error}")

        elif mode == "Spelling":
            try:
                with _circular_spinner("Checking spelling with Edit Distance..."):
                    _time_call(
                        "home_spellcheck",
                        run_spelling_correction,
                        app_state,
                        query,
                        int(option_values.get("max_distance", 2)),
                        int(option_values.get("candidate_topn", 5)),
                    )
            except Exception as error:
                st.error(f"Spelling correction failed: {type(error).__name__}: {error}")

        elif mode == "Word2Vec":
            try:
                st.session_state["w2v_similar_result"] = run_word2vec_similar(
                    app_state,
                    query,
                    int(option_values.get("topn", 10)),
                )
                st.session_state["w2v_similar_query"] = _normalize_word2vec_key(query)
            except Exception as error:
                st.error(f"Similarity failed: {type(error).__name__}: {error}")

        elif mode == "Semantic Expansion":
            try:
                with _circular_spinner("Expanding query with Word2Vec neighbors..."):
                    _time_call(
                        "home_semantic",
                        run_semantic_expansion,
                        app_state,
                        query,
                        int(option_values.get("topn_per_word", 3)),
                        float(option_values.get("min_similarity", 0.5)),
                    )
            except Exception as error:
                st.error(f"Semantic expansion failed: {type(error).__name__}: {error}")

        elif mode == "Porter":
            tokens = tokenize_words(query)
            st.session_state["porter_mini_result"] = {
                "input": query,
                "rows": [
                    {"Token": token, "Porter Stem": porter_stem(token)}
                    for token in tokens
                ],
            }

    if mode == "Keyword Search":
        _render_search_result(
            st.session_state.get("home_search_result"),
            "Retrieval Results",
            state=app_state,
            result_key="home_search",
        )

    elif mode == "Spelling":
        spell = app_state.get("spellcheck_result")
        if not spell:
            st.info("Enter a misspelled term and click Correct.")
        else:
            st.markdown("### Spelling Correction / Edit Distance")
            st.dataframe(_spell_correction_rows(spell), width="stretch", hide_index=True)
            st.caption(
                f"Original Query: {spell.get('original_query', '')} · "
                f"Suggested Correction: {spell.get('corrected_query', '')}"
            )
            corrected = spell.get("corrected_query", "")
            if corrected and st.button(
                "Search Corrected Query",
                key="home_search_corrected",
                type="primary",
            ):
                try:
                    with _circular_spinner("Searching corrected query..."):
                        st.session_state["spell_search_result"] = _time_call(
                            "home_corrected_search",
                            run_search,
                            app_state,
                            corrected,
                            "auto",
                        )
                        st.session_state["spell_search_page"] = 1
                except Exception as error:
                    st.error(f"Corrected search failed: {type(error).__name__}: {error}")
            if st.session_state.get("spell_search_result"):
                _render_search_result(
                    st.session_state["spell_search_result"],
                    "Corrected Query Retrieval",
                    state=app_state,
                    result_key="spell_search",
                )

    elif mode == "Word2Vec":
        if not (
            app_state.get("word2vec_result")
            and app_state.get("word2vec_result", {}).get("model") is not None
        ):
            st.info("The prepared Word2Vec model is not loaded.")
        else:
            _render_word2vec_home_result()

    elif mode == "Semantic Expansion":
        semantic = app_state.get("semantic_result")
        if not semantic:
            st.info("Enter a query and click Expand.")
        else:
            st.markdown("### Semantic Neighbors")
            st.dataframe(semantic.get("expansion_terms", []), width="stretch", hide_index=True)
            st.markdown("### Expanded Query")
            st.code(semantic.get("expanded_query", ""), language=None)
            with st.expander("Per-word Detail", expanded=False):
                for item in semantic.get("per_word", []) or []:
                    st.markdown(
                        f"**{item.get('word', '')}** — "
                        f"{'in vocabulary' if item.get('known') else 'OOV'}"
                    )
                    st.dataframe(item.get("expansions", []), width="stretch", hide_index=True)

            expanded_query = semantic.get("expanded_query", "")
            if expanded_query and st.button(
                "Search with Expanded Query",
                key="home_search_expanded",
                type="primary",
            ):
                try:
                    with _circular_spinner("Searching expanded query..."):
                        st.session_state["semantic_search_result"] = _time_call(
                            "home_semantic_search",
                            run_search,
                            app_state,
                            expanded_query,
                            "auto",
                        )
                        st.session_state["semantic_search_page"] = 1
                except Exception as error:
                    st.error(f"Expanded-query search failed: {type(error).__name__}: {error}")
            if st.session_state.get("semantic_search_result"):
                _render_search_result(
                    st.session_state["semantic_search_result"],
                    "Expanded Query Retrieval",
                    state=app_state,
                    result_key="semantic_search",
                )

    elif mode == "Porter":
        _render_porter_mini_result(st.session_state.get("porter_mini_result"))


# ============================================================
# STATEFUL PRESENTATION NAVIGATION
# ============================================================
# Streamlit's st.tabs() is visually convenient but does not expose a persistent
# selected-tab value. Any widget interaction reruns the script, so native tabs
# can fall back to the first tab after actions such as Search / Similar / Expand.
# These selectors keep the current page/subpage in st.session_state instead.

MAIN_PAGES = [
    "01 Retrieval",
    "02 Corpus Preparation",
    "03 Zipf Analysis",
    "04 CF-DF / TF-IDF",
    "05 Significant Words",
    "06 Optional Domains",
    "07 Diagnostics",
]

RETRIEVAL_MODES = [
    "Keyword Search",
    "Spelling",
    "Word2Vec",
    "Semantic Expansion",
    "Porter",
]


def _stateful_horizontal_selector(label, options, key):
    """Return a rerun-safe horizontal selection stored in session_state."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]

    segmented_control = getattr(st, "segmented_control", None)
    if callable(segmented_control):
        selected = segmented_control(
            label,
            options,
            selection_mode="single",
            key=key,
            label_visibility="collapsed",
        )
    else:
        selected = st.radio(
            label,
            options,
            horizontal=True,
            key=key,
            label_visibility="collapsed",
        )

    # A single-selection segmented control can theoretically return None if a
    # session is restored from an unusual browser state. Keep navigation valid.
    if selected not in options:
        return options[0]
    return selected


# ============================================================
# 6. PRESENTATION HEADER — V3
# ============================================================


formal_documents = int(summary.get("documents", 0) or 0)
unique_pmids = int(audit.get("unique_pmids", formal_documents) or formal_documents)
source_audit = audit.get("source_collection_audit", {}) or {}
validated_pool = int(
    audit.get("source_valid_pool_documents")
    or source_audit.get("valid_pool_documents")
    or formal_documents
)
system_glp1_audit = audit.get("system_glp1_representation_audit", {}) or {}
glp1_any_form_documents = (
    audit.get("system_glp1_any_form_documents")
    or system_glp1_audit.get("any_form_unique_documents")
    or audit.get("source_glp1_any_form_documents")
    or source_audit.get("any_requested_or_extra_form_unique_documents")
)

page_hint = st.session_state.get("active_main_page", MAIN_PAGES[0])
if page_hint not in MAIN_PAGES:
    page_hint = MAIN_PAGES[0]
    st.session_state["active_main_page"] = page_hint

if page_hint == "01 Retrieval":
    # Keep the same light/natural-green application canvas as the analysis pages.
    # Only the RETRIEVAL hero itself uses the dark green Project-1-inspired treatment.
    st.markdown(
        """
        <div class="retrieval-hero">
          <div class="retrieval-kicker">Artificial Intelligence Information Retrieval</div>
          <div class="retrieval-title">RETRIEVAL</div>
          <div class="retrieval-subtitle">Project #2 · Query Expansion Models</div>
          <div class="retrieval-flow">
            <span>Search</span><span>Spelling</span><span>Word2Vec</span><span>Semantic Expansion</span><span>Porter</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="p2-header">
          <div class="p2-kicker">Biomedical Information Retrieval</div>
          <div class="p2-title">Project 2</div>
          <div class="p2-subtitle">Zipf Analysis &amp; Query Expansion Models</div>
          <div class="p2-scope">1,000 English GLP-1 scientific abstracts · PubMed → direct PMC JATS · main abstract body only</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7. REPORT-ORDER FOLDER TABS — V3
# ============================================================


active_main_page = _stateful_horizontal_selector(
    "Main navigation",
    MAIN_PAGES,
    key="active_main_page",
)


# ============================================================
# TAB 01 — RETRIEVAL SYSTEM
# ============================================================


if active_main_page == "01 Retrieval":
    _render_retrieval_home()


# ============================================================
# TAB 02 — CORPUS PREPARATION (PART I–IV)
# ============================================================


if active_main_page == "02 Corpus Preparation":
    _section_header(
        "02",
        "Corpus Preparation",
        "Part I–IV: preprocessing, corpus statistics, Collection Frequency, and Document Frequency.",
    )
    st.caption("Part VI coverage on this page: Vocabulary size and High-frequency terms.")

    st.markdown("### A–D Processing Pipeline")
    _html_table(
        ["Condition", "Processing pipeline", "Purpose"],
        [
            ["A Basic", "Whitespace tokenization + case folding", "Baseline; punctuation can remain attached"],
            ["B Remove punctuation", "Punctuation-aware tokenizer + case folding", "Normalize token boundaries"],
            ["C Remove stopwords", "B + fixed English stopword removal", "Remove very common low-discrimination words"],
            ["D Porter stemming", "C + Porter stemmer", "Merge morphological variants"],
        ],
    )

    st.markdown("### A–D Numerical Comparison")
    _render_ad_summary_table(app_state)

    _render_preprocessing_summary_charts(app_state)

    st.caption(
        "A → B → C → D is cumulative by design. The same 1,000-document corpus is used for every condition."
    )

    st.markdown("### Top 50 Terms by CF — A / B / C / D")
    st.caption(
        "Part III — each condition is ranked independently by Collection Frequency (CF), "
        "with DF retained for Part IV. Use one shared Rank Range to compare the same "
        "positions across A / B / C / D."
    )

    rank_window_label = st.radio(
        "Rank Range",
        options=["1–10", "11–20", "21–30", "31–40", "41–50"],
        index=0,
        horizontal=True,
        key="v3_page02_cf_rank_range",
    )
    rank_windows = {
        "1–10": (1, 10),
        "11–20": (11, 20),
        "21–30": (21, 30),
        "31–40": (31, 40),
        "41–50": (41, 50),
    }
    rank_start, rank_end = rank_windows[rank_window_label]
    _render_cf_top50_four_conditions(
        app_state,
        rank_start=rank_start,
        rank_end=rank_end,
    )


# ============================================================
# TAB 03 — ZIPF REQUIRED ANALYSIS
# ============================================================


if active_main_page == "03 Zipf Analysis":
    _section_header("03", "Zipf Distribution & Required Analysis")

    prep_zipf = app_state.get("preprocessing_zipf_result") or {}
    by_condition = prep_zipf.get("conditions", {}) or {}

    if by_condition:
        # ① Overall numerical summary
        st.markdown("### A–D Zipf Numerical Comparison")
        _render_ad_summary_table(app_state)

        # ② Part V — Experiment 1
        _render_rank_frequency_four_condition_grids(app_state)

        # ③ Overall regression parameters belong directly with Experiment 2
        st.markdown("### Overall Regression Parameters — Slope + Intercept")
        _render_zipf_regression_parameters(app_state)
        st.caption(
            "log10(CF) = Intercept + Slope × log10(Rank)  ·  Zipf k = −Slope"
        )

        comparison = (app_state.get("preprocessing_zipf_result") or {}).get("comparison", [])
        if comparison:
            _plot_zipf_exponent_comparison(
                comparison,
                chart_key="v3_page03_zipf_exponent_comparison",
            )

        # ④ Part V — Experiment 2
        _render_loglog_four_condition_grids(app_state)

        # ⑤ Required Analysis Q4 — presentation-first visual summary
        st.markdown("### Q4 Evidence — Regional Fit Across A–D")
        _render_regional_r2_grouped_bar(app_state)
        st.markdown(
            '<div class="p2-callout"><b>Q4 conclusion:</b> Across A / B / C / D, the High-frequency region has the highest R² and therefore provides the best regional linear fit among the interpretable regions.</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Low-frequency region is not shown in the R² chart because many terms share the same minimum CF, "
            "producing near-zero variance in log10(CF); therefore R² is not informative for this region."
        )

        region_rows = _build_region_comparison_rows(app_state)
        if region_rows:
            with st.expander("View Detailed Regional Fit Table", expanded=False):
                _html_table(
                    ["Condition", "Region", "Rank Start", "Rank End", "Zipf k", "R²", "RMSE"],
                    region_rows,
                    align_right={2, 3, 4, 5, 6},
                )

        st.markdown(
            '<div class="p2-callout"><b>Q3 cue:</b> R² measures linear fit; discuss it together with curve shape, RMSE and regional fit rather than treating R² alone as proof.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Zipf analysis is not ready.")


# ============================================================
# TAB 04 — CF / DF / TF-IDF
# ============================================================


if active_main_page == "04 CF-DF / TF-IDF":
    _section_header("04", "CF vs DF & Connection to TF-IDF", "Primary required table uses Condition B; any term can also be compared across A–D.")

    stats = (app_state.get("term_statistics_by_condition", {}) or {}).get("B_no_punctuation")
    if stats:
        st.markdown(
            '<div class="p2-callout"><b>IDF convention in V3:</b> IDF(t) = log10(N / DF(t)), no smoothing. Zipf regression also uses log10.</div>',
            unsafe_allow_html=True,
        )
        _plot_cf_df(stats.get("rows", []), chart_key="v3_cfdf_B")

        # Part VII / VIII presentation selection:
        # use representative GLP-1 / biomedical content terms from the real
        # Condition B vocabulary instead of simply taking the CF Top-N list.
        # CF / DF are never hard-coded; every value below comes from stats["rows"].
        condition_b_rows = list(stats.get("rows", []) or [])
        rows_by_term = {
            str(row.get("term", "")).casefold().strip(): row
            for row in condition_b_rows
            if str(row.get("term", "")).strip()
        }

        representative_terms = [
            "glp-1",
            "receptor",
            "glucagon-like",
            "agonists",
            "diabetes",
            "obesity",
            "glucose",
            "insulin",
            "weight",
            "treatment",
            "semaglutide",
            "liraglutide",
            "study",
            "patients",
            "placebo",
            "randomized",
            "effects",
            "cardiovascular",
            "risk",
            "secretion",
        ]

        # Only aliases that map to an actually existing Condition B term may be used.
        # They are fallbacks for spelling / singular-plural variants, never invented data.
        representative_aliases = {
            "agonists": ["agonist"],
            "patients": ["patient"],
            "randomized": ["randomised", "randomization", "randomisation"],
            "effects": ["effect"],
            "cardiovascular": ["cardiac"],
            "secretion": ["secretory"],
            "study": ["studies"],
        }

        fallback_content_terms = [
            "glycemic",
            "glycaemic",
            "clinical",
            "metabolic",
            "peptide",
            "cells",
            "levels",
            "increased",
            "loss",
            "therapy",
            "hormone",
            "control",
            "body",
            "type",
        ]

        selected_rows = []
        selected_terms = set()
        resolved_term = {}

        def _append_existing_term(requested_term):
            candidates = [requested_term] + representative_aliases.get(requested_term, [])
            for candidate in candidates:
                key = str(candidate).casefold().strip()
                row = rows_by_term.get(key)
                actual_term = str((row or {}).get("term", "")).casefold().strip()
                if row and actual_term and actual_term not in selected_terms:
                    selected_rows.append(row)
                    selected_terms.add(actual_term)
                    resolved_term[requested_term] = actual_term
                    return True
            return False

        for term in representative_terms:
            _append_existing_term(term)

        # If an exact preferred term is absent, backfill only with real content terms
        # that exist in the Condition B vocabulary so the Part VII table remains >=20.
        for term in fallback_content_terms:
            if len(selected_rows) >= 20:
                break
            _append_existing_term(term)

        if len(selected_rows) < 20:
            common_function_words = {
                "and", "the", "of", "in", "to", "a", "with", "for", "that",
                "is", "were", "was", "are", "as", "by", "on", "or", "from",
                "at", "be", "their", "not", "an", "have", "has", "this", "these",
            }
            for row in sorted(
                condition_b_rows,
                key=lambda item: (
                    -int(item.get("collection_frequency", 0) or 0),
                    str(item.get("term", "")).casefold(),
                ),
            ):
                term = str(row.get("term", "")).casefold().strip()
                if not term or term in selected_terms or term in common_function_words:
                    continue
                selected_rows.append(row)
                selected_terms.add(term)
                if len(selected_rows) >= 20:
                    break

        selected_rows = selected_rows[:20]
        cf_df_required = [
            {
                "Term": row.get("term"),
                "CF": int(row.get("collection_frequency", 0) or 0),
                "DF": int(row.get("document_frequency", 0) or 0),
            }
            for row in selected_rows
        ]

        # Part VIII intentionally reuses a subset of the same Part VII terms.
        idf_preferred_terms = [
            "glp-1",
            "receptor",
            "glucagon-like",
            "diabetes",
            "weight",
            "glucose",
            "semaglutide",
            "liraglutide",
            "study",
            "cardiovascular",
        ]
        selected_part7_by_term = {
            str(row.get("term", "")).casefold().strip(): row
            for row in selected_rows
        }

        idf_rows_selected = []
        idf_seen = set()
        for requested_term in idf_preferred_terms:
            candidate_terms = [requested_term] + representative_aliases.get(requested_term, [])
            for candidate in candidate_terms:
                key = str(candidate).casefold().strip()
                row = selected_part7_by_term.get(key)
                if row and key not in idf_seen:
                    idf_rows_selected.append(row)
                    idf_seen.add(key)
                    break

        # Guarantee the required 10 rows using only terms already shown in Part VII.
        if len(idf_rows_selected) < 10:
            for row in selected_rows:
                key = str(row.get("term", "")).casefold().strip()
                if key in idf_seen:
                    continue
                idf_rows_selected.append(row)
                idf_seen.add(key)
                if len(idf_rows_selected) >= 10:
                    break

        total_documents = int(stats.get("total_documents", 0) or 0)
        idf_required = []
        for row in idf_rows_selected[:10]:
            df = int(row.get("document_frequency", 0) or 0)
            idf_value = (
                math.log10(total_documents / df)
                if total_documents > 0 and df > 0
                else None
            )
            idf_required.append(
                {
                    "Term": row.get("term"),
                    "CF": int(row.get("collection_frequency", 0) or 0),
                    "DF": df,
                    "IDF": round(idf_value, 5) if idf_value is not None else None,
                }
            )

        left, right = st.columns(2)
        with left:
            st.markdown(f"### CF vs DF — Representative 20 Terms ({len(cf_df_required)} shown)")
            st.dataframe(cf_df_required, width="stretch", hide_index=True, height=560)
            st.caption(
                "Part VII uses representative GLP-1 / biomedical content terms from the actual "
                "Condition B vocabulary so different CF / DF relationships and document coverage are visible."
            )
        with right:
            st.markdown(f"### IDF / TF-IDF Connection — Representative 10 Terms ({len(idf_required)} shown)")
            st.dataframe(idf_required, width="stretch", hide_index=True, height=350)
            st.caption(
                "Part VIII reuses terms from the Part VII table. IDF is computed dynamically as "
                "log10(N / DF) using the same Condition B corpus statistics."
            )
            st.markdown(
                '<div class="p2-callout"><b>TF-IDF link:</b> TF-IDF(t,d) = TF(t,d) × IDF(t). The 10-term table above connects corpus-level DF / IDF to the document-level TF-IDF inspector below.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Inspect Any Term Across A–D")
        term_query = st.text_input("Term", value="glp-1", key="v3_term_inspect")
        _render_term_across_conditions(app_state, term_query)

        with st.expander("Optional document-level TF-IDF inspector", expanded=False):
            document_ids = list((app_state.get("index_data") or {}).get("documents", {}).keys())
            if document_ids:
                selected_document = st.selectbox("Document ID", document_ids, key="v3_tfidf_doc")
                top_n = st.slider("Top TF-IDF terms", 10, 100, 30, 10, key="v3_tfidf_topn")
                if st.button("Compute Document TF-IDF", key="v3_tfidf_button"):
                    st.session_state["tfidf_result"] = _time_call(
                        "document_tfidf",
                        run_document_tfidf_analysis,
                        app_state,
                        selected_document,
                        None,
                        10,
                        False,
                        int(top_n),
                    )
                tfidf = st.session_state.get("tfidf_result")
                if tfidf:
                    st.dataframe(tfidf.get("rows", []), width="stretch", hide_index=True)
    else:
        st.info("CF/DF/IDF results are not ready.")


# ============================================================
# TAB 05 — SIGNIFICANT WORDS / RESOLVING POWER
# ============================================================


if active_main_page == "05 Significant Words":
    _section_header(
        "05",
        "Significant Words & Resolving Power",
        "Bridge from Zipf-like word frequency to retrieval value.",
    )

    st.markdown(
        '<div class="p2-callout"><b>The most frequent words are not necessarily the most descriptive.</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Page 03 establishes the formal Zipf analysis. This page asks a different IR question: "
        "**given that frequency is highly uneven, which terms may actually help distinguish documents?**"
    )

    if not app_state.get("significant_words_by_condition"):
        if st.button("Compute Significant Words Analysis", key="v3_significant_compute", type="primary"):
            _time_call("significant_words", run_significant_words_analysis, app_state, 20)
            st.rerun()

    significant_results = app_state.get("significant_words_by_condition") or {}
    if significant_results:
        condition_keys = [
            "A_basic",
            "B_no_punctuation",
            "C_no_stopwords",
            "D_porter_stemming",
        ]
        available_keys = [key for key in condition_keys if significant_results.get(key)]

        selected_key = st.selectbox(
            "Preprocessing condition",
            available_keys,
            index=(available_keys.index("B_no_punctuation") if "B_no_punctuation" in available_keys else 0),
            format_func=_condition_short_label,
            key="v3_significant_condition",
            help="Condition B is the default main view; switch A/B/C/D during the presentation to compare how preprocessing changes the curve and candidate terms.",
        )

        selected_result = significant_results.get(selected_key) or {}
        term_count = int(selected_result.get("term_count", 0) or 0)

        st.markdown(
            "**Actual frequency curve:** measured CF from the frozen corpus.  "
            "**Conceptual Resolving-Power Curve:** a lecture-style `low → high → low` hump, "
            "dynamically parameterized by the illustrative Upper / Lower cut-offs."
        )
        st.caption(
            "This is a conceptual visualization only, dynamically parameterized by the "
            "illustrative cut-offs; it is not measured from corpus data."
        )

        cutoff_enabled = st.toggle(
            "Show movable illustrative Upper / Lower cut-offs",
            value=True,
            key=f"v3_significant_cutoff_enabled_{selected_key}",
        )
        cutoff_range = None
        if cutoff_enabled and term_count >= 2:
            default_upper = max(1, int(round(term_count * 0.05)))
            default_lower = max(default_upper + 1, int(round(term_count * 0.25)))
            default_lower = min(term_count, default_lower)
            cutoff_range = st.slider(
                "Illustrative cut-off positions by rank — drag both handles during the demo",
                min_value=1,
                max_value=term_count,
                value=(default_upper, default_lower),
                step=1,
                key=f"v3_significant_cutoff_range_{selected_key}",
            )
            st.caption(
                "Conceptual cut-offs only. The course material does not provide a fixed numerical "
                "cut-off formula. Initial slider positions (5% and 25% of this vocabulary) are merely "
                "presentation defaults with no analytical status; move them freely during the demo."
            )

        _plot_significant_frequency_and_concept(
            app_state,
            selected_key,
            show_cutoffs=cutoff_enabled,
            cutoff_range=cutoff_range,
        )

        st.caption(
            "Conceptual Resolving-Power Curve: illustrative lecture / Luhn–van Rijsbergen concept. "
            "Moving the cut-offs changes the hump position and width, but the curve remains a "
            "conceptual visualization rather than a corpus measurement."
        )

        region_cols = st.columns(3)
        with region_cols[0]:
            st.markdown("#### Very High Frequency")
            st.markdown(
                "Common / ubiquitous terms can dominate CF while contributing relatively little "
                "document discrimination."
            )
        with region_cols[1]:
            st.markdown("#### Middle Frequency")
            st.markdown(
                "Potential significant words: repeated enough to provide evidence while still "
                "retaining cross-document discrimination."
            )
        with region_cols[2]:
            st.markdown("#### Very Low Frequency / Long Tail")
            st.markdown(
                "Rare terms may be specific, but rarity alone does not guarantee stable or useful "
                "retrieval evidence."
            )

        st.markdown("### Candidate Significant Terms from This Corpus")
        if cutoff_enabled and cutoff_range:
            st.caption(
                "The table shows real corpus terms whose Rank falls inside the current illustrative "
                "Upper / Lower cut-off window. Rows are ordered by Rank; no exploratory proxy score "
                "is used on this page."
            )
        else:
            st.caption(
                "No cut-off filter is active; the table shows the first ranked corpus terms for the "
                "selected condition. Columns are real Rank / CF / DF / IDF statistics."
            )

        candidate_rows = _candidate_significant_rows(
            app_state,
            selected_key,
            cutoff_range=(cutoff_range if cutoff_enabled else None),
            top_n=50,
        )
        st.dataframe(candidate_rows, width="stretch", hide_index=True)

        st.markdown(
            '<div class="p2-callout"><b>Interpretation boundary:</b> Upper / Lower cut-offs on this page are user-controlled conceptual annotations. They are not Zipf Regional Fit boundaries, not professor-defined numerical thresholds, and not Luhn-defined thresholds.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Run the V3 preparation pipeline to populate the Significant Words analysis.")


# ============================================================
# TAB 06 — OPTIONAL TWO-DOMAIN CHALLENGE
# ============================================================


if active_main_page == "06 Optional Domains":
    _section_header("06", "Optional Challenge — Compare Two Domains", "Optional only; it does not block the required Project 2 presentation.")

    domain_result = app_state.get("domain_comparison_result")
    if domain_result:
        st.dataframe(domain_result.get("comparison_rows", []), width="stretch", hide_index=True)
        left, right = st.columns(2)
        with left:
            a = domain_result.get("domain_a", {})
            st.markdown(f"### {a.get('label', 'Domain A')} — Top 10")
            st.dataframe(a.get("top_terms", []), width="stretch", hide_index=True)
            _plot_loglog_with_regression(
                a.get("zipf", {}),
                title=f"{a.get('label', 'Domain A')} Zipf",
                chart_key="v3_domain_a_zipf",
            )
        with right:
            b = domain_result.get("domain_b", {})
            st.markdown(f"### {b.get('label', 'Domain B')} — Top 10")
            st.dataframe(b.get("top_terms", []), width="stretch", hide_index=True)
            _plot_loglog_with_regression(
                b.get("zipf", {}),
                title=f"{b.get('label', 'Domain B')} Zipf",
                chart_key="v3_domain_b_zipf",
            )
    else:
        st.info("No second-domain corpus is prepared yet. Main required Project 2 results are unaffected.")
        with st.expander("Run from a prepared local Domain B JATS folder", expanded=False):
            other_path = st.text_input(
                "Domain B JATS folder",
                value="",
                placeholder=r"D:\...\data\optional_domain_b_jats_xml",
                key="v3_domain_b_path",
            )
            label_b = st.text_input("Domain B label", value="Computer Science", key="v3_domain_b_label")
            if st.button("Run Optional Comparison", key="v3_domain_compare", type="primary"):
                if not other_path.strip():
                    st.warning("Enter a Domain B folder path.")
                else:
                    try:
                        with _circular_spinner("Loading Domain B and comparing 500 vs 500..."):
                            _time_call(
                                "optional_domain",
                                run_optional_domain_comparison_from_local_jats,
                                app_state,
                                other_path,
                                label_b,
                                "C_no_stopwords",
                                500,
                                10,
                            )
                        st.rerun()
                    except Exception as error:
                        st.error(f"Optional comparison failed: {type(error).__name__}: {error}")


# ============================================================
# TAB 07 — DIAGNOSTICS / REPRODUCIBILITY
# ============================================================


if active_main_page == "07 Diagnostics":
    _section_header("07", "System / Reproducibility / Diagnostics", "Engineering details stay outside the main presentation path.")

    metadata_rows = [
        {"Item": "Data mode", "Value": str(app_state.get("data_mode", ""))},
        {"Item": "Official raw directory", "Value": str(paths.get("raw_glp1_jats_dir", ""))},
        {"Item": "Validated pool", "Value": f"{validated_pool:,}"},
        {"Item": "Experiment documents", "Value": f"{summary.get('documents', 0):,}"},
        {"Item": "Corpus SHA-256", "Value": str(app_state.get("corpus_sha256", ""))},
        {"Item": "IDF convention", "Value": "log10(N / DF), no smoothing"},
        {"Item": "Runtime cache", "Value": str(paths.get("prepared_runtime_exists", False))},
        {"Item": "Analysis scope", "Value": str(audit.get("analysis_scope", "main abstract body only"))},
        {"Item": "Prepared generated UTC", "Value": str(app_state.get("prepared_metadata", {}).get("generated_at_utc", ""))},
    ]
    st.dataframe(metadata_rows, width="stretch", hide_index=True)

    with st.expander("Dataset audit", expanded=False):
        st.json(audit)

    st.markdown("### Document Statistics")
    document_rows = get_document_stats(app_state)
    st.dataframe(document_rows, width="stretch", hide_index=True, height=420)
    st.download_button(
        "Download Document Statistics CSV",
        data=_csv_bytes(document_rows),
        file_name="project2_v3_document_statistics.csv",
        mime="text/csv",
    )

    st.markdown("### Timing")
    timing_rows = [
        {"Task": key, "Seconds": round(float(value), 6), "Display": _format_seconds(value)}
        for key, value in st.session_state["timings"].items()
    ]
    st.dataframe(timing_rows, width="stretch", hide_index=True)

    st.markdown("### Errors / Warnings")
    if app_state.get("build_errors"):
        st.error(f"{len(app_state['build_errors'])} build/load warnings")
        st.json(app_state.get("build_errors"))
    else:
        st.success("No build/load errors recorded.")

    export_payload = {
        "dataset_audit": audit,
        "corpus_sha256": app_state.get("corpus_sha256", ""),
        "index_summary": summary,
        "idf_convention": "log10(N / DF), no smoothing",
        "preprocessing_comparison": (app_state.get("preprocessing_zipf_result") or {}).get("comparison", []),
        "significant_words": app_state.get("significant_words_by_condition", {}),
        "porter_comparison": (app_state.get("porter_zipf_result") or {}).get("comparison", {}),
        "word2vec_metadata": (app_state.get("word2vec_result") or {}).get("metadata", {}),
        "optional_domain_comparison": app_state.get("domain_comparison_result"),
    }
    st.download_button(
        "Download Report Summary JSON",
        data=_safe_json_bytes(export_payload),
        file_name="project2_report_summary_v3.json",
        mime="application/json",
    )
