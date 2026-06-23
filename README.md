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

### Hugging Face token (required for Task 3+)

Task 3's generator calls the Hugging Face Inference API, which needs a free token:

1. Create one at https://huggingface.co/settings/tokens (read access is enough).
2. Some models (e.g. Mistral) require accepting the license on the model's page first.
3. Set it as an environment variable before running anything in `src/` or `notebooks/03_*`:

```powershell
# Windows PowerShell
$env:HF_TOKEN = "hf_xxxxxxxx"
```
```bash
# Mac/Linux
export HF_TOKEN=hf_xxxxxxxx
```

Never commit a real token — `.env` is already gitignored if you'd rather keep it there with `python-dotenv`.

## Data

Download the data from the links provided in the challenge doc and place them here:

| File | Destination | Used in |
|------|-------------|---------|
| Full CFPB complaint dataset | `data/raw/` | Task 1 |
| `complaint_embeddings.parquet` (pre-built embeddings) | `data/raw/` | Tasks 3–4 |

## Tasks

- **Task 1 — EDA & Preprocessing:** `notebooks/01_eda_preprocessing.ipynb` — explore and clean the complaint data, save to `data/processed/filtered_complaints.csv`.
- **Task 2 — Chunking, Embedding, Indexing:** `src/build_vector_store.py` — stratified sample, chunk, embed, and persist a ChromaDB vector store.
- **Task 3 — RAG Core Logic & Evaluation:** `src/` (`vector_index.py`, `retriever.py`, `prompt_template.py`, `generator.py`, `rag_pipeline.py`) + `notebooks/03_rag_evaluation.ipynb`. Run in order:
  ```bash
  python src/build_faiss_index.py   # indexes the pre-built complaint_embeddings.parquet (no re-embedding)
  jupyter notebook notebooks/03_rag_evaluation.ipynb   # runs the pipeline against test questions, exports the eval table
  ```
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
