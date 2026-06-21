"""
Gradio chat interface for the CrediTrust complaint RAG chatbot.

This is a placeholder scaffold for Task 4. Once src/rag_pipeline.py exposes a
function like `answer_question(question: str) -> dict` (returning the answer
and the retrieved source chunks), wire it in below.
"""

import gradio as gr

# from src.rag_pipeline import answer_question  # uncomment once implemented


def answer_question_stub(question: str):
    """Placeholder until the real RAG pipeline (Task 3) is implemented."""
    answer = "RAG pipeline not yet implemented. This is a placeholder response."
    sources = ["[source chunk 1 placeholder]", "[source chunk 2 placeholder]"]
    return answer, "\n\n---\n\n".join(sources)


def clear_chat():
    return "", "", ""


with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown("# CrediTrust Complaint Assistant")
    gr.Markdown(
        "Ask a plain-English question about customer complaints "
        "(e.g. *Why are people unhappy with Credit Cards?*)"
    )

    question_box = gr.Textbox(label="Your question", placeholder="Ask something...")

    with gr.Row():
        submit_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear")

    answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)
    sources_box = gr.Textbox(label="Retrieved sources", lines=8, interactive=False)

    submit_btn.click(
        fn=answer_question_stub,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[question_box, answer_box, sources_box],
    )

if __name__ == "__main__":
    demo.launch()
