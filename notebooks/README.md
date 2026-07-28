# CrediTrust Complaint Assistant
### A RAG-powered chatbot that turns 9.6 million customer complaints into instant, sourced answers

[![CI](https://github.com/lielina-pro/rag-complaint-chatbot/actions/workflows/unittests.yml/badge.svg)](https://github.com/lielina-pro/rag-complaint-chatbot/actions/workflows/unittests.yml)
![Python](https://img.shields.io/badge/python-3.14-blue)
![Tests](https://img.shields.io/badge/tests-56%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🏦 Business Problem

Financial services companies receive hundreds of thousands of customer complaints every year — and the teams responsible for acting on them have no fast way to find patterns across all that unstructured text. A compliance analyst trying to understand why customers are frustrated with credit card fees currently has to either read through complaints manually, wait for a quarterly report, or submit a data request and wait days for a response.

**This chatbot changes that.** Any team member — in Product, Support, or Compliance — can type a plain-English question and get a direct, sourced answer in seconds, grounded in real complaint narratives from the CFPB consumer database. Every answer shows exactly which complaints it came from, so you can verify it yourself rather than just trust the model.

---

## 🔍 Solution Overview

The CrediTrust Complaint Assistant is a full Retrieval-Augmented Generation (RAG) pipeline built on top of the Consumer Financial Protection Bureau's (CFPB) public complaint database. Rather than fine-tuning a model on complaint data (expensive, slow, and hard to update), it retrieves the most relevant real complaint narratives for any question and hands them to an LLM to synthesize a structured answer — combining the factual reliability of a search engine with the readability of a language model.

**How it works:**

```
User question
     │
     ▼
Embed question (all-MiniLM-L6-v2)
     │
     ▼
FAISS similarity search → top-k complaint chunks (1.37M indexed)
     │
     ▼
Inject chunks into prompt template (financial analyst persona)
     │
     ▼
Generate answer (DeepSeek-V3-0324 via HF Inference API)
     │
     ▼
Stream answer + show cited sources
```

---

## 📊 Key Results

| Metric | Result |
|---|---|
| Raw CFPB complaints processed | 9,609,797 |
| Complaints with usable narrative text | 480,539 (5.0% of raw) |
| Chunks indexed in FAISS vector store | 1,375,327 |
| Embedding dimensions | 384 (all-MiniLM-L6-v2) |
| Average answer quality score | 4 / 5 across 8 evaluation questions |
| End-to-end response time (CPU, no GPU) | ~10 seconds |
| Unit tests passing | 56 / 56 |
| CI pipeline | ✅ GitHub Actions on every push and PR |

**Products covered:** Credit Card · Personal Loan · Savings Account · Money Transfer

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A free [Hugging Face account](https://huggingface.co) and access token
- The `complaint_embeddings.parquet` file (pre-built, ~2GB — link in project notes)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/lielina-pro/rag-complaint-chatbot.git
cd rag-complaint-chatbot

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Hugging Face token
# Create a .env file in the project root (already gitignored):
echo HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxx > .env
```

### Build the vector store

```bash
# This indexes the 1.37M pre-built embeddings — takes ~90 seconds, no re-embedding needed
python src/build_faiss_index.py
```

### Run the app

```bash
python app.py
# Opens at http://127.0.0.1:7860
```

### Run the tests

```bash
python -m pytest tests/ -q
# Expected: 56 passed
```

---

## 🖥️ Demo

**Empty interface — ready for a question:**

![CrediTrust Complaint Assistant UI](docs/demo_ui.png)

**Live answer with cited sources:**

![CrediTrust Complaint Assistant — answer with sources](docs/demo_answer.png)

The answer above was generated from real CFPB complaint narratives. Notice the structured breakdown (Perceived Unfair Denials, High Interest Rates, Lack of Transparency, Poor Customer Service) with inline source references — every point can be traced back to an actual complaint in the database.

> 💡 **Try these questions to get started:**
> - *"Why are people unhappy with Credit Cards?"*
> - *"What are the most common complaints about Personal Loans?"*
> - *"Are there complaints about unauthorized transactions?"*
> - *"Do customers complain about poor customer service?"*

---

## 📁 Project Structure

```
rag-complaint-chatbot/
│
├── app.py                          # Gradio interface — streaming answers + sources
│
├── src/
│   ├── build_faiss_index.py        # CLI: streams complaint_embeddings.parquet → FAISS index
│   ├── vector_index.py             # Memory-safe streaming indexer + load/save utilities
│   ├── retriever.py                # Embeds queries + searches FAISS for top-k chunks
│   ├── embedding.py                # Wrapper around sentence-transformers/all-MiniLM-L6-v2
│   ├── prompt_template.py          # Financial analyst prompt with context injection
│   ├── generator.py                # HF Inference API client (DeepSeek-V3-0324)
│   ├── rag_pipeline.py             # Orchestrates retrieve → prompt → generate → stream
│   ├── build_vector_store.py       # Task 2: builds a sample ChromaDB store from scratch
│   ├── chunking.py                 # RecursiveCharacterTextSplitter wrapper
│   ├── sampling.py                 # Stratified sampling with proportion preservation
│   └── __init__.py
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb  # Task 1: EDA + cleaning across full 9.6M-row dataset
│   └── 03_rag_evaluation.ipynb     # Task 3: qualitative evaluation across 8 test questions
│
├── tests/
│   ├── test_sampling.py            # Stratified sampling correctness + edge cases
│   ├── test_chunking.py            # Chunking logic, empty text, metadata passthrough
│   ├── test_vector_index.py        # FAISS build/load, streaming, ChromaDB-schema detection
│   ├── test_retriever.py           # Retrieval correctness with fake embedder
│   ├── test_prompt_template.py     # Prompt structure + context injection
│   ├── test_generator.py           # Generator with fake HF client
│   ├── test_rag_pipeline.py        # End-to-end pipeline with fakes
│   ├── test_app.py                 # Gradio UI logic + error handling
│   └── test_sanity.py
│
├── data/
│   ├── raw/                        # complaint_embeddings.parquet (gitignored — ~2GB)
│   └── processed/                  # filtered_complaints.csv (gitignored)
│
├── vector_store/                   # full_dataset.faiss + metadata (gitignored — ~2GB)
├── reports/                        # task3_evaluation_table.md
├── notebooks/
├── .github/workflows/
│   └── unittests.yml               # CI: runs pytest on every push + PR
├── requirements.txt
├── .env                            # HF_TOKEN — gitignored, never committed
└── .gitignore
```

---

## ⚙️ Technical Details

### Data Pipeline
- **Source:** CFPB Consumer Complaint Database — 9,609,797 raw complaints
- **Filtering:** Mapped 21 raw product labels → 4 target categories; kept only records with a narrative of ≥ 3 words after cleaning
- **Text cleaning:** Lowercase, removed CFPB redaction placeholders (`XXXX`, `XX/XX/XXXX`), stripped boilerplate openers
- **Final dataset:** 480,539 complaints

### Chunking & Embedding
- **Splitter:** LangChain `RecursiveCharacterTextSplitter` — 500 char chunks, 50 char overlap; tries paragraph/sentence boundaries before hard character cuts
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dim, ~80MB, runs on CPU, normalized to cosine similarity
- **Task 2 sample:** 12,000 complaints → 35,178 chunks → ChromaDB (for exploration)

### Vector Store
- **Full dataset:** `complaint_embeddings.parquet` (1,375,327 pre-built chunks)
- **Index type:** `faiss.IndexFlatIP` (exact, brute-force, inner product = cosine similarity on normalized vectors)
- **Memory strategy:** Parquet streamed in batches of 50,000 rows; metadata written incrementally via `pyarrow.ParquetWriter` — peak RAM stays near one batch + the index (~2GB), avoiding the 4GB single-allocation crash that flat `pd.read_parquet` would cause on the full file
- **Build time:** ~90 seconds on CPU

### Generation
- **Model:** `deepseek-ai/DeepSeek-V3-0324` via Hugging Face Inference API
- **Prompt design:** Financial analyst persona; context-only instruction ("do not use prior knowledge"); instructs the model to explicitly say when the answer isn't supported by the sources
- **Streaming:** Answers stream token-by-token in the UI via Gradio's generator function support

### Retrieval Evaluation
Eight representative questions were run through the pipeline and scored qualitatively (1–5 scale: accuracy, groundedness, completeness). Average score: **4 / 5**. Full evaluation table in `reports/task3_evaluation_table.md`.

---

## 🔧 Planned Improvements (Week 12)

- [ ] **Interactive analytics dashboard** — complaint volume by product over time, top issues per category, retrieval quality metrics (second Gradio tab or Streamlit page)
- [ ] **Full type hint coverage** across `src/` modules + `RAGConfig` dataclass to consolidate scattered configuration defaults
- [ ] **Deployment** to Hugging Face Spaces — live demo link, no local setup required
- [ ] **Blog post** — technical write-up covering the architecture, key engineering decisions, and lessons learned

---

## 👤 Author

**Lielina Fekadu**
[LinkedIn](https://www.linkedin.com/in/lielina-fekadu-993b54362) · [GitHub](https://github.com/lielina-pro)

Built as part of **10 Academy's AI Mastery (KAIM9) program** — Week 7 capstone project, enhanced in Week 12.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
