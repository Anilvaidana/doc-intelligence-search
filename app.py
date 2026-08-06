"""
=============================================================================
ENTERPRISE DOCUMENT INTELLIGENCE & SEMANTIC SEARCH — STREAMLIT APP
=============================================================================
A real, runnable RAG (Retrieval-Augmented Generation) pipeline:

  Upload docs -> Chunk -> Embed (transformer) -> FAISS ANN index
  -> Semantic search on your question -> (optional) LLM answer

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

First run downloads a small embedding model (~80MB) from Hugging Face,
so you need normal internet access the first time you run it.
=============================================================================
"""

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit command
# -----------------------------------------------------------------------
st.set_page_config(page_title="Document Intelligence Search", layout="wide")


# -----------------------------------------------------------------------
# STAGE 3: EMBEDDING MODEL (the deep learning core of the whole system)
# -----------------------------------------------------------------------
# @st.cache_resource means Streamlit loads this ONCE per server session,
# not on every user interaction — otherwise it would reload a ~80MB model
# every time someone typed a letter into the search box.
#
# "all-MiniLM-L6-v2" is a small transformer bi-encoder: it reads a piece
# of text through several self-attention layers and pools the result into
# one 384-dimensional vector. It was trained with contrastive learning,
# so semantically similar sentences land close together in that
# 384-dimensional space — even if they don't share any words.
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------------------------------------
# STAGE 2: CHUNKING
# -----------------------------------------------------------------------
# We can't embed an entire document as one vector (too much information
# gets averaged away, and it's too long for most models' context window).
# So we split each document into overlapping word windows. The overlap
# (50 words) means an idea that falls right on a chunk boundary still
# appears fully in at least one chunk.
def chunk_text(text, source_name, chunk_size=120, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append({"text": " ".join(chunk_words), "source": source_name})
        if end >= len(words):
            break
        start = end - overlap  # step forward, but re-include the overlap
    return chunks


# -----------------------------------------------------------------------
# STAGE 4: ANN INDEX (FAISS)
# -----------------------------------------------------------------------
# IndexFlatIP = inner product search. Since our embeddings are
# L2-normalized (unit length), the inner product between two vectors is
# mathematically identical to cosine similarity. This is the SIMPLEST
# FAISS index (still an exact/brute-force scan, not truly "approximate").
# At enterprise scale (millions of chunks) you'd swap this one line for
# faiss.IndexHNSWFlat(dim, 32) or an IVF index — everything else in this
# file stays the same, which is the whole point of using FAISS: the ANN
# algorithm is swappable behind one line of code.
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index


# -----------------------------------------------------------------------
# STAGE 5: SEMANTIC SEARCH
# -----------------------------------------------------------------------
def search(query, model, index, chunks, k=3):
    # Embed the query with the SAME model used for the documents — query
    # and documents must live in the same vector space to be comparable.
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vector, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({"score": float(score), **chunks[idx]})
    return results


# -----------------------------------------------------------------------
# STAGE 7: ANSWER SYNTHESIS (RAG) — optional, needs an Anthropic API key
# -----------------------------------------------------------------------
# We hand the LLM ONLY the retrieved chunks and instruct it to answer
# strictly from them. This "grounding" is what makes RAG answers
# trustworthy — the model isn't guessing from memory, it's reading the
# same passages the user could read themselves.
def synthesize_answer(query, results, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    context = "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite the source file for your answer. "
        "If the answer isn't contained in the context, say so plainly.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# -----------------------------------------------------------------------
# BUILT-IN SAMPLE DOCUMENTS — lets a user try the app with zero setup
# -----------------------------------------------------------------------
SAMPLE_DOCS = {
    "hr_policy.txt": (
        "Remote Work Policy. Employees may work from home up to three days "
        "per week with manager approval. Home office equipment purchases up "
        "to $500 per year are reimbursable with receipts submitted through "
        "the expense portal within 30 days of purchase. "
        "Parental Leave Policy. Full-time employees are entitled to 16 weeks "
        "of paid parental leave following the birth or adoption of a child."
    ),
    "security_guidelines.txt": (
        "Data Classification Standard. All company data must be classified "
        "as Public, Internal, Confidential, or Restricted. Restricted data "
        "must be encrypted at rest using AES-256. "
        "Incident Response. Any suspected security breach must be reported "
        "to the security team within one hour of discovery via the incident "
        "hotline or the security-incidents Slack channel."
    ),
    "vendor_contract_acme.txt": (
        "Payment Terms. Acme Corp shall invoice monthly in arrears. Payment "
        "is due within Net 45 days of invoice receipt. "
        "Termination Clause. Either party may terminate this agreement with "
        "90 days written notice. Upon termination, Acme Corp shall provide "
        "a full data export within 30 days."
    ),
}


# =========================================================================
# STREAMLIT UI
# =========================================================================
st.title("Enterprise Document Intelligence & Semantic Search")
st.caption("Upload documents, then ask questions in plain English — retrieval is by meaning, not keywords.")

# --- Sidebar: setup controls ---
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Anthropic API key (optional — enables AI-written answers)",
        type="password",
        help="Leave blank to see raw retrieved passages only, no generated answer.",
    )
    use_sample = st.checkbox("Use built-in sample documents", value=True)
    uploaded_files = st.file_uploader(
        "...or upload your own .txt files", accept_multiple_files=True, type=["txt"]
    )
    build_clicked = st.button("Build / rebuild index", type="primary")

# Session state persists the index across reruns (Streamlit reruns the
# whole script on every interaction, so without this we'd lose the index
# every time the user typed in the search box).
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

model = load_embedding_model()

if build_clicked:
    # Gather source documents: sample docs and/or uploaded files
    documents = {}
    if use_sample:
        documents.update(SAMPLE_DOCS)
    for f in uploaded_files or []:
        documents[f.name] = f.read().decode("utf-8")

    if not documents:
        st.sidebar.error("No documents to index — check a sample or upload a file.")
    else:
        # STAGE 2: chunk every document
        all_chunks = []
        for name, text in documents.items():
            all_chunks.extend(chunk_text(text, name))

        # STAGE 3: embed every chunk
        with st.spinner(f"Embedding {len(all_chunks)} chunks..."):
            embeddings = model.encode(
                [c["text"] for c in all_chunks], normalize_embeddings=True
            )

        # STAGE 4: build the ANN index
        index = build_index(np.array(embeddings))

        st.session_state.index = index
        st.session_state.chunks = all_chunks
        st.sidebar.success(f"Indexed {len(all_chunks)} chunks from {len(documents)} document(s)")

# --- Main panel: search ---
query = st.text_input("Ask a question about your documents", placeholder="e.g. Can I get my home office chair paid for?")

if query:
    if st.session_state.index is None:
        st.warning("Click 'Build / rebuild index' in the sidebar first.")
    else:
        # STAGE 5: semantic search
        results = search(query, model, st.session_state.index, st.session_state.chunks, k=3)

        st.subheader("Top matching passages")
        for r in results:
            st.markdown(f"**{r['source']}**  ·  similarity `{r['score']:.3f}`")
            st.write(r["text"])
            st.divider()

        # STAGE 7: optional LLM-generated grounded answer
        if api_key:
            with st.spinner("Generating grounded answer..."):
                try:
                    answer = synthesize_answer(query, results, api_key)
                    st.subheader("AI answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Couldn't generate an answer: {e}")
        else:
            st.info("Add an Anthropic API key in the sidebar to also get a synthesized, cited answer.")
