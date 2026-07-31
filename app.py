"""
CrediTrust Complaint Assistant — Gradio UI (Task 4 / Week 12 redesign)

Professional fintech interface with:
- Custom CSS design system (white + teal accent)
- DM Serif Display title, Inter body
- Source cards with teal left border
- k-slider tucked in collapsed Advanced accordion
- Empty states that invite action instead of floating blank headers
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)  # .env always wins, even over a stale shell env var

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.embedding import Embedder
from src.generator import Generator
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_index import load_index_and_metadata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INDEX_PATH = str(Path(__file__).resolve().parent / "vector_store" / "full_dataset.faiss")
METADATA_PATH = str(Path(__file__).resolve().parent / "vector_store" / "full_dataset_metadata.parquet")

# ---------------------------------------------------------------------------
# Pipeline init (graceful — UI still launches if setup is incomplete)
# ---------------------------------------------------------------------------
pipeline: RAGPipeline | None = None
INIT_ERROR: str | None = None
metadata_df: pd.DataFrame | None = None


def build_pipeline() -> tuple[RAGPipeline, pd.DataFrame]:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{INDEX_PATH}'. "
            "Run `python src/build_faiss_index.py` first."
        )
    index, meta_df = load_index_and_metadata(INDEX_PATH, METADATA_PATH)
    embedder = Embedder()
    retriever = Retriever(index, meta_df, embedder)
    generator = Generator()
    return RAGPipeline(retriever, generator), meta_df


try:
    pipeline, metadata_df = build_pipeline()
except Exception as exc:
    INIT_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
PRODUCT_COLORS = {
    "Credit Card":      "#00B4D8",
    "Personal Loan":    "#F4A261",
    "Savings Account":  "#56CFE1",
    "Money Transfer":   "#9D4EDD",
}

PRODUCT_ICONS = {
    "Credit Card":      "💳",
    "Personal Loan":    "🏦",
    "Savings Account":  "🏧",
    "Money Transfer":   "💸",
}


def render_markdown_bold(text: str) -> str:
    """Convert **bold** markdown to <strong> HTML tags for display."""
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)


def format_sources(chunks: list[dict]) -> str:
    if not chunks:
        return "_No sources retrieved._"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        product = chunk.get("product_category", "Unknown")
        company = chunk.get("company", "")
        score = chunk.get("score")
        text = (chunk.get("chunk_text") or chunk.get("document") or "").strip()
        if len(text) > 500:
            text = text[:497] + "..."   # three dots — not Unicode ellipsis
        score_str = f"· relevance {score:.2f}" if score is not None else ""
        icon = PRODUCT_ICONS.get(product, "📄")
        color = PRODUCT_COLORS.get(product, "#0891B2")
        company_str = f" · {company}" if company else ""
        parts.append(
            f'<div class="source-card" style="border-left-color:{color}">'
            f'<div class="source-header">'
            f'<span class="source-num">Source {i}</span>'
            f'<span class="source-tag" style="background:{color}22;color:{color}">'
            f'{icon} {product}</span>'
            f'<span class="source-meta">{company_str} {score_str}</span>'
            f'</div>'
            f'<div class="source-text">{text}</div>'
            f'</div>'
        )
    return "\n".join(parts)


def clear_all():
    return "", "", ""


def ask_question(question: str, k: int):
    question = (question or "").strip()

    if not question:
        yield (
            '<div class="empty-state">✏️ Please enter a question above, '
            'or click one of the examples.</div>',
            ""
        )
        return

    if INIT_ERROR:
        yield (
            f'<div class="error-state">⚠️ The app isn\'t fully set up yet.<br><br>'
            f'<code>{INIT_ERROR}</code></div>',
            ""
        )
        return

    yield '<div class="generating">⟳ Searching 1.37 million complaint records…</div>', ""

    try:
        full_answer = ""
        sources_html = ""

        for chunk in pipeline.answer_stream(question, k=k):
            if isinstance(chunk, tuple):
                text_chunk, raw_sources = chunk
                if raw_sources:
                    sources_html = format_sources(
                        raw_sources if isinstance(raw_sources, list)
                        else raw_sources.get("sources", [])
                    )
                if text_chunk:
                    full_answer = text_chunk
                    rendered = render_markdown_bold(full_answer)
                    yield (
                        f'<div class="answer-text">{rendered}<span class="cursor">▌</span></div>',
                        sources_html
                    )
            elif isinstance(chunk, dict):
                sources_html = format_sources(chunk.get("sources", []))
            else:
                full_answer += chunk
                rendered = render_markdown_bold(full_answer)
                yield (
                    f'<div class="answer-text">{rendered}<span class="cursor">▌</span></div>',
                    sources_html
                )

        rendered = render_markdown_bold(full_answer)
        yield f'<div class="answer-text">{rendered}</div>', sources_html

    except Exception as exc:
        yield (
            f'<div class="error-state">⚠️ Something went wrong while generating the answer:'
            f'<br><br><code>{exc}</code></div>',
            ""
        )

# ---------------------------------------------------------------------------
# Analytics dashboard helpers
# ---------------------------------------------------------------------------
CHART_COLORS = ["#0891B2", "#D97706", "#56CFE1", "#7C3AED"]
CHART_BG     = "#FFFFFF"
CHART_GRID   = "#F1F5F9"
CHART_TEXT   = "#0F172A"
CHART_MUTED  = "#64748B"
PRODUCTS     = ["Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.facecolor":   CHART_BG,
    "figure.facecolor": CHART_BG,
    "axes.edgecolor":   "#E2E8F0",
    "axes.labelcolor":  CHART_TEXT,
    "xtick.color":      CHART_MUTED,
    "ytick.color":      CHART_MUTED,
    "text.color":       CHART_TEXT,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": CHART_GRID,
    "grid.linewidth": 0.8,
})


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def chart_product_volume(df: pd.DataFrame) -> plt.Figure:
    col = _col(df, "product_category")
    if col is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5, "product_category column not found", ha="center", va="center")
        return fig
    counts = df[col].value_counts()
    counts = counts.reindex([p for p in PRODUCTS if p in counts.index])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(counts))]
    bars = ax.barh(counts.index, counts.values, color=colors, height=0.55)
    ax.set_xlabel("Number of complaint chunks", fontsize=9, color=CHART_MUTED)
    ax.set_title("Complaint Volume by Product", fontsize=11, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + counts.values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8.5, color=CHART_MUTED)
    fig.tight_layout()
    return fig


def chart_top_companies(df: pd.DataFrame) -> plt.Figure:
    col = _col(df, "company")
    if col is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5, "company column not found", ha="center", va="center")
        return fig
    counts = df[col].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=CHART_COLORS[0], height=0.55)
    ax.set_xlabel("Number of complaint chunks", fontsize=9, color=CHART_MUTED)
    ax.set_title("Top 10 Companies by Complaint Volume", fontsize=11, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + counts.values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8, color=CHART_MUTED)
    fig.tight_layout()
    return fig


def chart_top_issues(df: pd.DataFrame) -> plt.Figure:
    prod_col  = _col(df, "product_category")
    issue_col = _col(df, "issue")
    if prod_col is None or issue_col is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Required columns not found", ha="center", va="center")
        return fig
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Top 5 Issues per Product Category", fontsize=12, fontweight="bold", y=1.01)
    axes_flat = axes.flatten()
    valid_prods = [p for p in PRODUCTS if p in df[prod_col].values]
    for i, product in enumerate(valid_prods):
        ax = axes_flat[i]
        subset = df[df[prod_col] == product]
        issues = (
            subset[issue_col].dropna().str.strip()
            .replace("", pd.NA).dropna()
            .value_counts().head(5)
        )
        color = CHART_COLORS[i % len(CHART_COLORS)]
        ax.barh(issues.index[::-1], issues.values[::-1], color=color, height=0.55)
        ax.set_title(product, fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("Chunks", fontsize=8, color=CHART_MUTED)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K" if x >= 1000 else str(int(x))))
        wrapped = [lb[:30] + "..." if len(lb) > 30 else lb for lb in issues.index[::-1]]
        ax.set_yticks(range(len(wrapped)))
        ax.set_yticklabels(wrapped, fontsize=7.5)
    for j in range(len(valid_prods), 4):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    return fig


def chart_state_distribution(df: pd.DataFrame) -> plt.Figure:
    col = _col(df, "state")
    if col is None:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "state column not found", ha="center", va="center")
        return fig
    counts = df[col].dropna().value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    bar_colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(counts))]
    ax.bar(counts.index, counts.values, color=bar_colors, width=0.6)
    ax.set_title("Top 15 US States by Complaint Volume", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Complaint chunks", fontsize=9, color=CHART_MUTED)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    fig.tight_layout()
    return fig


def build_kpi_html(df: pd.DataFrame) -> str:
    prod_col  = _col(df, "product_category")
    comp_col  = _col(df, "company")
    state_col = _col(df, "state")
    total     = f"{len(df):,}"
    n_prod    = str(df[prod_col].nunique()) if prod_col  else "4"
    n_comp    = f"{df[comp_col].nunique():,}" if comp_col else "—"
    n_state   = str(df[state_col].nunique()) if state_col else "—"
    cards = [
        ("🗄", "Total Chunks Indexed",   total,   "#0891B2"),
        ("📦", "Product Categories",     n_prod,  "#D97706"),
        ("🏢", "Unique Companies",       n_comp,  "#7C3AED"),
        ("🗺", "US States Represented",  n_state, "#16A34A"),
    ]
    html = ""
    for icon, label, value, color in cards:
        html += (
            f'<div class="kpi-card">'
            f'<div class="kpi-icon" style="color:{color}">{icon}</div>'
            f'<div class="kpi-value" style="color:{color}">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>'
        )
    return f'<div class="kpi-row">{html}</div>'


# ---------------------------------------------------------------------------
# CSS design system
# ---------------------------------------------------------------------------
CSS = """
/* ── Tokens ─────────────────────────────────────────────── */
:root {
    --bg:           #F8FAFC;
    --surface:      #FFFFFF;
    --surface-2:    #F1F5F9;
    --border:       #E2E8F0;
    --teal:         #0891B2;
    --teal-dim:     #0891B222;
    --amber:        #D97706;
    --text:         #0F172A;
    --text-muted:   #64748B;
    --radius:       10px;
    --radius-sm:    6px;
    --font-display: 'DM Serif Display', Georgia, serif;
    --font-body:    'Inter', system-ui, sans-serif;
}
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600&display=swap');
body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}
.app-header { padding: 36px 0 28px; border-bottom: 1px solid var(--border); margin-bottom: 32px; }
.app-title { font-family: var(--font-display) !important; font-size: 2.4rem !important; font-weight: 400 !important; color: var(--text) !important; letter-spacing: -0.5px; margin: 0 0 8px; line-height: 1.2; }
.app-subtitle { color: var(--text-muted); font-size: 0.95rem; line-height: 1.6; max-width: 640px; }
.teal { color: var(--teal); }
.status-pill { display: inline-flex; align-items: center; gap: 6px; background: #00B4D811; border: 1px solid #00B4D833; border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; color: var(--teal); margin-top: 12px; font-weight: 500; }
.status-dot { width: 7px; height: 7px; background: var(--teal); border-radius: 50%; animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.8); } }
.input-section label { color: #334155 !important; font-size: 0.82rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; margin-bottom: 8px !important; }
.input-section textarea, .input-section input[type="text"] { background: #FFFFFF !important; border: 1.5px solid var(--border) !important; border-radius: var(--radius) !important; color: var(--text) !important; font-family: var(--font-body) !important; font-size: 1.05rem !important; padding: 14px 16px !important; resize: none !important; transition: border-color 0.2s, box-shadow 0.2s; box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important; }
.input-section textarea:focus, .input-section input[type="text"]:focus { border-color: var(--teal) !important; outline: none !important; box-shadow: 0 0 0 3px var(--teal-dim) !important; }
.input-section .block, .input-section .wrap, .input-section > div { background: transparent !important; border: none !important; padding: 0 !important; box-shadow: none !important; }
.ask-btn { background: var(--teal) !important; color: #0F1923 !important; font-weight: 600 !important; font-size: 1rem !important; border: none !important; border-radius: var(--radius) !important; padding: 14px 32px !important; cursor: pointer !important; transition: background 0.2s, transform 0.1s !important; width: 100% !important; letter-spacing: 0.02em !important; }
.ask-btn:hover { background: #00CFE8 !important; transform: translateY(-1px) !important; }
.ask-btn:active { transform: translateY(0) !important; }
.clear-btn { background: transparent !important; border: 1px solid var(--border) !important; color: var(--text-muted) !important; border-radius: var(--radius) !important; padding: 10px 20px !important; font-size: 0.9rem !important; cursor: pointer !important; transition: border-color 0.2s, color 0.2s !important; width: 100% !important; }
.clear-btn:hover { border-color: var(--text-muted) !important; color: var(--text) !important; }
.panel-label { color: #334155; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px; display: flex; align-items: center; gap: 10px; }
.panel-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.answer-text { color: var(--text); font-size: 1rem; line-height: 1.85; white-space: pre-wrap; max-width: 680px; }
.answer-text strong { color: var(--text); font-weight: 600; }
.cursor { color: var(--teal); animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
.generating { color: var(--text-muted); font-size: 0.9rem; animation: fade-pulse 1.4s ease-in-out infinite; }
@keyframes fade-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }
.empty-state { color: var(--text-muted); font-size: 0.95rem; text-align: center; padding: 32px 0; font-style: italic; }
.error-state { color: #DC2626; font-size: 0.9rem; line-height: 1.6; }
.error-state code { background: #FEE2E2; border-radius: 4px; padding: 2px 6px; font-size: 0.82rem; }
.source-card { background: var(--surface-2); border: 1px solid var(--border); border-left: 3px solid var(--teal); border-radius: var(--radius-sm); padding: 14px 16px; margin-bottom: 10px; }
.source-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
.source-num { color: var(--text-muted); font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.source-tag { border-radius: 4px; padding: 2px 8px; font-size: 0.78rem; font-weight: 500; }
.source-meta { color: var(--text-muted); font-size: 0.78rem; margin-left: auto; }
.source-text { color: #475569; font-size: 0.88rem; line-height: 1.6; }
#advanced-accordion { background: #FFFFFF !important; border: 1.5px solid var(--border) !important; border-radius: var(--radius-sm) !important; margin-top: 12px !important; }
#advanced-accordion > .label-wrap, #advanced-accordion summary { color: #475569 !important; font-size: 0.85rem !important; font-weight: 500 !important; padding: 10px 14px !important; background: #FFFFFF !important; }
#advanced-accordion .block { background: #FFFFFF !important; border: none !important; box-shadow: none !important; }
input[type="range"] { accent-color: var(--teal) !important; }
label, .gr-label, span.svelte-1b6s6s, .label-wrap span { color: #334155 !important; font-weight: 600 !important; }
.gr-samples-table td, .gr-sample-textbox, table.samples td, .samples button { background: #FFFFFF !important; border: 1.5px solid #CBD5E1 !important; border-radius: 6px !important; color: #334155 !important; font-size: 0.88rem !important; transition: border-color 0.2s, color 0.2s !important; }
.gr-samples-table td:hover, .gr-sample-textbox:hover, table.samples td:hover, .samples button:hover { border-color: var(--teal) !important; color: var(--teal) !important; }
.app-footer { border-top: 1px solid var(--border); padding-top: 20px; margin-top: 40px; color: #475569; font-size: 0.82rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
.footer-link { color: var(--teal) !important; text-decoration: none; font-weight: 600; }
.footer-link:hover { text-decoration: underline; }
.footer-stat { background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 4px 12px; font-size: 0.78rem; color: #475569; }
.divider { height: 1px; background: var(--border); margin: 28px 0; }
footer, .built-with, #footer { display: none !important; }
.gradio-container .gap { gap: 0 !important; }
.input-section > .block { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
.kpi-row { display: flex; gap: 16px; margin: 20px 0 28px; flex-wrap: wrap; }
.kpi-card { flex: 1; min-width: 140px; background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 10px; padding: 18px 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.kpi-icon  { font-size: 1.4rem; margin-bottom: 6px; }
.kpi-value { font-size: 1.55rem; font-weight: 700; line-height: 1.2; }
.kpi-label { font-size: 0.76rem; color: #64748B; margin-top: 5px; font-weight: 500; }
.analytics-header { padding: 8px 0 16px; }
.analytics-title  { font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 4px; }
.analytics-sub    { font-size: 0.87rem; color: #64748B; }
.tab-nav button { font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 0.92rem !important; color: #64748B !important; border-bottom: 2px solid transparent !important; background: transparent !important; padding: 10px 20px !important; }
.tab-nav button.selected { color: #0891B2 !important; border-bottom-color: #0891B2 !important; }
"""

# ---------------------------------------------------------------------------
# Example questions
# ---------------------------------------------------------------------------
EXAMPLES = [
    "Why are people unhappy with Credit Cards?",
    "What are the most common complaints about Personal Loans?",
    "What issues do customers report with Savings Accounts?",
    "What problems are customers experiencing with Money Transfers?",
    "Are there complaints about unauthorized transactions?",
    "Do customers complain about poor customer service?",
    "What billing or fee-related issues appear across products?",
    "Are there any complaints related to fraud?",
]

STATUS_OK = (
    '<div class="status-pill"><span class="status-dot"></span>'
    '1,375,327 complaints indexed · Ready</div>'
)
STATUS_ERROR = (
    '<div class="status-pill" style="border-color:#FC818133;color:#FC8181;background:#FC818111">'
    '⚠ Setup required</div>'
)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
_kpi_html = build_kpi_html(metadata_df) if metadata_df is not None else ""

with gr.Blocks(css=CSS, title="CrediTrust Complaint Assistant") as demo:

    gr.HTML(f"""
    <div class="app-header">
        <div class="app-title">CrediTrust <span class="teal">Complaint</span> Assistant</div>
        <div class="app-subtitle">
            AI-powered complaint analysis across Credit Cards, Personal Loans,
            Savings Accounts, and Money Transfers — grounded in 1.37 million
            real customer complaint narratives.
        </div>
        {STATUS_OK if not INIT_ERROR else STATUS_ERROR}
    </div>
    """)

    with gr.Tabs(elem_classes=["tab-nav"]):

        with gr.Tab("💬  Ask a Question"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["input-section"]):
                    question_box = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. Why are people unhappy with Credit Cards?",
                        lines=3, max_lines=6,
                    )
                    with gr.Row():
                        ask_btn   = gr.Button("Ask →", elem_classes=["ask-btn"], variant="primary")
                        clear_btn = gr.Button("Clear",  elem_classes=["clear-btn"])
                    with gr.Accordion("⚙  Advanced settings", open=False, elem_id="advanced-accordion"):
                        k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1,
                                             label="Number of source chunks to retrieve (k)")
                    gr.HTML('<div class="divider"></div>')
                    gr.Examples(examples=EXAMPLES, inputs=question_box,
                                label="Example questions — click to try")

                with gr.Column(scale=7):
                    gr.HTML('<div class="panel-label">Answer</div>')
                    answer_box = gr.HTML(
                        value='<div class="empty-state">Your answer will appear here.</div>')
                    gr.HTML('<div class="divider"></div>')
                    gr.HTML('<div class="panel-label">Sources</div>')
                    sources_box = gr.HTML(
                        value='<div class="empty-state">Retrieved complaint excerpts will appear here.</div>')

        with gr.Tab("📊  Analytics"):
            gr.HTML("""
            <div class="analytics-header">
                <div class="analytics-title">Complaint Data Overview</div>
                <div class="analytics-sub">
                    Aggregated from 1,375,327 complaint chunks indexed across
                    4 product categories — all derived from the CFPB Consumer Complaint Database.
                </div>
            </div>
            """)
            gr.HTML(_kpi_html)
            gr.HTML('<div class="divider"></div>')
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-label">Volume by Product</div>')
                    gr.Plot(value=chart_product_volume(metadata_df) if metadata_df is not None else None, show_label=False)
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-label">Top 10 Companies</div>')
                    gr.Plot(value=chart_top_companies(metadata_df) if metadata_df is not None else None, show_label=False)
            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="panel-label">Top 5 Issues per Product Category</div>')
            gr.Plot(value=chart_top_issues(metadata_df) if metadata_df is not None else None, show_label=False)
            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="panel-label">Geographic Distribution (Top 15 States)</div>')
            gr.Plot(value=chart_state_distribution(metadata_df) if metadata_df is not None else None, show_label=False)

    gr.HTML("""
    <div class="app-footer">
        <span>
            Built by <a class="footer-link"
            href="https://www.linkedin.com/in/lielina-fekadu-993b54362"
            target="_blank">Lielina Fekadu</a>
            · 10 Academy KAIM9
        </span>
        <span><span class="footer-stat">🗄 FAISS · all-MiniLM-L6-v2 · DeepSeek-V3</span></span>
    </div>
    """)

    ask_btn.click(fn=ask_question, inputs=[question_box, k_slider], outputs=[answer_box, sources_box])
    question_box.submit(fn=ask_question, inputs=[question_box, k_slider], outputs=[answer_box, sources_box])
    clear_btn.click(fn=clear_all, outputs=[question_box, answer_box, sources_box])

if __name__ == "__main__":
    demo.launch()
