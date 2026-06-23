"""
Task 4 — Interactive chat interface for the CrediTrust complaint RAG chatbot.

Wires together everything from Tasks 2-3:
  - vector_store/full_dataset.faiss + full_dataset_metadata.parquet (build
    these first with `python src/build_faiss_index.py`)
  - src/retriever.py, src/prompt_template.py, src/generator.py, src/rag_pipeline.py

Run with:
    python app.py
"""

from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)  # .env always wins, even over a stale HF_TOKEN already set in this shell session

from src.embedding import Embedder
from src.generator import Generator
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_index import load_index_and_metadata

INDEX_PATH = os.path.join("vector_store", "full_dataset.faiss")
METADATA_PATH = os.path.join("vector_store", "full_dataset_metadata.parquet")
DEFAULT_K = 5

EXAMPLE_QUESTIONS = [
    "Why are people unhappy with Credit Cards?",
    "What are the most common complaints about Personal Loans?",
    "What issues do customers report with Savings Accounts?",
    "What problems are customers experiencing with Money Transfers?",
    "Are there complaints about unauthorized transactions?",
    "Do customers complain about poor customer service?",
    "What billing or fee-related issues appear across products?",
    "Are there any complaints related to fraud?",
]


# ---------------------------------------------------------------------------
# Build the pipeline once at startup. If anything's missing (the index
# hasn't been built yet, or HF_TOKEN isn't set), don't crash the whole app --
# capture the error and surface it inside the UI itself, since that's a far
# more useful message for a non-technical user than a terminal traceback.
# ---------------------------------------------------------------------------
def build_pipeline():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)):
        raise FileNotFoundError(
            f"Vector store not found at '{INDEX_PATH}'. Run "
            "`python src/build_faiss_index.py` first (after downloading "
            "complaint_embeddings.parquet into data/raw/)."
        )
    index, metadata_df = load_index_and_metadata(INDEX_PATH, METADATA_PATH)
    embedder = Embedder()
    retriever = Retriever(index, metadata_df, embedder)
    generator = Generator()  # reads HF_TOKEN from the environment
    return RAGPipeline(retriever, generator, k=DEFAULT_K)


try:
    pipeline = build_pipeline()
    INIT_ERROR = None
except Exception as e:  # noqa: BLE001 -- intentionally broad, surfaced in the UI
    pipeline = None
    INIT_ERROR = str(e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_sources(chunks: list[dict]) -> str:
    """Render retrieved chunks as a readable Markdown block, so the user can
    verify the answer against the actual complaint text it came from."""
    if not chunks:
        return "_No sources retrieved._"

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        product = chunk.get("product_category", "Unknown product")
        company = chunk.get("company")
        score = chunk.get("score")
        text = (chunk.get("chunk_text") or "").strip()
        if len(text) > 350:
            text = text[:350].rstrip() + "..."

        header = f"**Source {i} — {product}**"
        if company:
            header += f" · {company}"
        if score is not None:
            header += f" · relevance {score:.2f}"

        lines.append(f"{header}\n\n> {text}")

    return "\n\n---\n\n".join(lines)


def ask_question(question: str, k: int):
    """Generator function -- Gradio streams each yielded value to the UI as
    it arrives, which is what makes the token-by-token answer display work."""
    question = (question or "").strip()

    if not question:
        yield "Please enter a question above, or pick one of the examples below.", ""
        return

    if INIT_ERROR:
        yield (
            f"⚠️ The app isn't fully set up yet:\n\n{INIT_ERROR}",
            "",
        )
        return

    try:
        sources_md = ""
        for partial_answer, chunks in pipeline.answer_stream(question, k=int(k)):
            sources_md = format_sources(chunks)
            yield partial_answer, sources_md
    except Exception as e:  # noqa: BLE001 -- show the user *something* rather than crash
        yield f"Something went wrong while generating the answer:\n\n`{e}`", ""


def clear_all():
    return "", "", ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
)

CUSTOM_CSS = """
.app-title { margin-bottom: 0 !important; }
.app-subtitle { color: #5b6472; margin-top: 0.25rem !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown("# 💬 CrediTrust Complaint Assistant", elem_classes="app-title")
    gr.Markdown(
        "Ask a plain-English question about customer complaints across Credit Cards, "
        "Personal Loans, Savings Accounts, and Money Transfers. Answers are generated "
        "from real complaint narratives — every answer shows the sources it came from "
        "so you can verify it yourself.",
        elem_classes="app-subtitle",
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. Why are people unhappy with Credit Cards?",
            scale=4,
            autofocus=True,
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        k_slider = gr.Slider(
            minimum=1, maximum=10, value=DEFAULT_K, step=1,
            label="Number of source chunks to retrieve (k)",
        )
        clear_btn = gr.Button("Clear", scale=0)

    gr.Examples(
        examples=EXAMPLE_QUESTIONS,
        inputs=question_box,
        label="Example questions",
    )

    gr.Markdown("### Answer")
    answer_md = gr.Markdown()

    gr.Markdown("### Sources")
    sources_md = gr.Markdown()

    ask_btn.click(fn=ask_question, inputs=[question_box, k_slider], outputs=[answer_md, sources_md])
    question_box.submit(fn=ask_question, inputs=[question_box, k_slider], outputs=[answer_md, sources_md])
    clear_btn.click(fn=clear_all, outputs=[question_box, answer_md, sources_md])

    if INIT_ERROR:
        gr.Markdown(f"⚠️ **Setup needed:** {INIT_ERROR}")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
