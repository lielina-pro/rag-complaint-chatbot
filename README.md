# RAG Complaint Chatbot — CrediTrust Financial

Intelligent complaint analysis tool for CrediTrust Financial. Turns raw, unstructured
customer complaint narratives into actionable insight via a Retrieval-Augmented
Generation (RAG) chatbot, so Product, Support, and Compliance teams can ask plain-English
questions and get evidence-backed answers in seconds.

## Project Structure

```
rag-complaint-chatbot/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── raw/                # original CFPB dataset (not committed)
│   └── processed/           # cleaned/filtered dataset (not committed)
├── vector_store/            # persisted FAISS/ChromaDB index (not committed)
├── notebooks/                # EDA and exploration notebooks
├── src/                      # pipeline source code
├── tests/                    # unit tests
├── app.py                    # Gradio/Streamlit interface
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Data

Download the data from the links provided in the challenge doc and place them here:

| File | Destination | Used in |
|------|-------------|---------|
| Full CFPB complaint dataset | `data/raw/` | Task 1 |
| `complaint_embeddings.parquet` (pre-built embeddings) | `data/raw/` | Tasks 3–4 |

## Tasks

- **Task 1 — EDA & Preprocessing:** `notebooks/` — explore and clean the complaint data, save to `data/processed/filtered_complaints.csv`.
- **Task 2 — Chunking, Embedding, Indexing:** `src/` — build a stratified sample, chunk narratives, embed, and persist a vector store.
- **Task 3 — RAG Core Logic & Evaluation:** `src/` — retriever + prompt template + generator, evaluated qualitatively on 5–10 test questions.
- **Task 4 — Interactive UI:** `app.py` — Gradio/Streamlit chat interface with source display.

## Running the app

```bash
python app.py
# or, if using Streamlit:
streamlit run app.py
```

## Running tests

```bash
pytest tests/
```
