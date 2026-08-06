# Enterprise Document Intelligence & Semantic Search System

A Retrieval-Augmented Generation (RAG) system that lets users search enterprise documents — policies, contracts, spreadsheets — using natural language instead of exact keywords. Ask a question in plain English and get back the actual relevant passage, even if it shares no words with your query.

**Live demo:** `[add your deployed Streamlit / Cloudflare tunnel link here]`

---

## Why this exists

Traditional keyword search (`Ctrl+F`, SQL `LIKE`) fails when the user's words don't match the document's words. Asking *"can I get my home office chair paid for?"* should surface a reimbursement policy even though the word "chair" never appears in it. This system retrieves by **meaning**, not string matching, using dense vector embeddings and approximate nearest neighbor search.

---

## Architecture

```
Documents (.txt / .csv / .xlsx)
        │
        ▼
   Chunking            — split into overlapping passages / serialize table rows
        │
        ▼
   Embedding            — transformer bi-encoder (all-MiniLM-L6-v2) → 384-dim vectors
        │
        ▼
   ANN Index (FAISS)    — fast similarity search over vectors
        │
        ▼
User query → embed → semantic search → top-k matching passages
        │
        ▼
   (optional) LLM answer synthesis — Claude generates a grounded, cited answer
        │
        ▼
   Streamlit UI
```

---

## Features

- Upload `.txt`, `.csv`, or `.xlsx` documents — spreadsheet rows are serialized with column context before embedding
- Semantic search: retrieves by meaning, not keyword overlap
- Optional grounded answer generation via the Anthropic API, with source citation
- Runs entirely locally except the optional LLM call — no data leaves your machine for search/retrieval
- Deployable in minutes via Google Colab + Cloudflare Tunnel, or Streamlit Community Cloud

---

## Tech stack

| Layer | Technology |
|---|---|
| Embedding model | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector index / ANN | FAISS (`IndexFlatIP`, swappable for `IndexHNSWFlat` at scale) |
| Answer generation | Anthropic Claude API (optional) |
| Tabular parsing | pandas / openpyxl |
| UI | Streamlit |
| Tunneling (Colab) | Cloudflare Tunnel |

---

## Setup

### Option A — Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```
Opens automatically at `http://localhost:8501`.

### Option B — Run in Google Colab (no local install)

1. Open a new Colab notebook
2. Paste the contents of `app.py` into a cell using `%%writefile app.py`
3. Run:
   ```python
   !pip install -q streamlit sentence-transformers faiss-cpu anthropic pandas openpyxl
   !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
   !chmod +x cloudflared-linux-amd64
   ```
4. Launch and tunnel:
   ```python
   import subprocess, time
   subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])
   time.sleep(10)
   subprocess.Popen(["./cloudflared-linux-amd64", "tunnel", "--url", "http://localhost:8501"])
   time.sleep(8)
   ```
5. Check the tunnel output for a public `https://*.trycloudflare.com` URL

### Option C — Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select this repo → main file: `app.py` → **Deploy**

---

## Usage

1. Check **"Use built-in sample documents"** (or upload your own `.txt` / `.csv` / `.xlsx` files) in the sidebar
2. Click **Build / rebuild index**
3. Type a question in plain English
4. View the top matching passages, ranked by similarity score
5. (Optional) Add an Anthropic API key in the sidebar to also get a synthesized, cited answer instead of raw passages

---

## Project structure

```
.
├── app.py              # Streamlit application (full pipeline)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Evaluation notes

- Retrieval quality depends heavily on chunk size/overlap and the embedding model used; `all-MiniLM-L6-v2` is small and fast but a larger model (`all-mpnet-base-v2`, or a domain fine-tuned model) improves accuracy at the cost of latency.
- `IndexFlatIP` performs an exact brute-force search — fine for small/medium document sets. At enterprise scale (millions of chunks), swap in `IndexHNSWFlat` or an IVF index for true Approximate Nearest Neighbor speed with a small, tunable recall trade-off.
- No re-ranking stage is included in the base version; adding a cross-encoder re-ranker on the top-k results would improve precision, particularly for queries that are lexically close but semantically distinct.

## Limitations

- Row-by-row tabular embedding works for lookup-style questions ("what are Acme's payment terms?") but not for aggregate/numeric questions ("what's our total spend with Acme this year?") — those need a computation path (e.g. text-to-SQL or a pandas agent), not vector search.
- No access-control/permissions layer — a production deployment would need document-level filtering so users only retrieve content they're authorized to see.
- OCR for scanned PDFs is not implemented in this version; only text-native files are supported.

## Future work

- Cross-encoder re-ranking stage
- Hybrid search (BM25 + dense retrieval) for better handling of exact-match queries (IDs, names, codes)
- Access control filtering at query time
- OCR pipeline for scanned documents
- Recall@k / NDCG evaluation harness comparing ANN index types

---

## License

MIT — see `LICENSE` file.
