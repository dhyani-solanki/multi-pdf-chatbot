# chatapp.py
import streamlit as st
from io import BytesIO
from typing import List, Dict
import os
import traceback

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# ----------------------------
# Streamlit page config & helpers
# ----------------------------
st.set_page_config(page_title="📚 Multi-PDF Chat", layout="wide")
st.title("📚 Multi-PDF Chat — Local (FAISS + local LLMs)")

# Create storage dir for faiss
INDEX_DIR = "faiss_index"
os.makedirs(".", exist_ok=True)


# ----------------------------
# Caching heavy resources
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # cached SentenceTransformerEmbeddings instance
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_hf_pipeline(model_name: str):
    """
    Return a Hugging Face text2text/text-generation pipeline wrapped for LangChain.
    Use device=-1 to force CPU.
    """
    # Use "text2text-generation" for T5-like models
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
        device=-1,  # CPU; set to 0 for GPU if available and configured
        max_length=512,
    )
    # Wrap into a LangChain LLM wrapper
    return HuggingFacePipeline(pipeline=pipe)


# ----------------------------
# PDF loading and chunking
# ----------------------------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """
    Read PDF bytes and return concatenated text.
    Uses PyPDF2 PdfReader (works for uploaded Streamlit files).
    """
    text = ""
    try:
        stream = BytesIO(file_bytes)
        reader = PdfReader(stream)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # basic cleanup: collapse multiple newlines
                text += page_text + "\n\n"
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")
    return text


def split_text_to_chunks(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


# ----------------------------
# Vector store helpers
# ----------------------------
def create_or_append_faiss(text_chunks_with_meta: List[Dict]):
    """
    text_chunks_with_meta: list of {"text": "...", "metadata": {...}}
    If index exists, append; otherwise create new FAISS index and save.
    """
    embeddings = get_embeddings()
    if os.path.exists(INDEX_DIR) and len(os.listdir(INDEX_DIR)) > 0:
        # load and append
        try:
            vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            texts = [it["text"] for it in text_chunks_with_meta]
            metas = [it["metadata"] for it in text_chunks_with_meta]
            vs.add_texts(texts, metadatas=metas)
            vs.save_local(INDEX_DIR)
            return vs
        except Exception as e:
            # if loading fails, remove dir and recreate
            st.warning("Existing index could not be loaded (corrupt?). Rebuilding from scratch.")
            try:
                for f in os.listdir(INDEX_DIR):
                    os.remove(os.path.join(INDEX_DIR, f))
            except Exception:
                pass

    # create new
    texts = [it["text"] for it in text_chunks_with_meta]
    metas = [it["metadata"] for it in text_chunks_with_meta]
    vs = FAISS.from_texts(texts, embedding=get_embeddings(), metadatas=metas)
    vs.save_local(INDEX_DIR)
    return vs


def load_vector_store_safe():
    embeddings = get_embeddings()
    if os.path.exists(INDEX_DIR) and len(os.listdir(INDEX_DIR)) > 0:
        try:
            vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            return vs
        except Exception as e:
            st.warning("Failed to load saved index: " + str(e))
            return None
    return None


# ----------------------------
# LLM + prompt
# ----------------------------
def build_llm_chain(model_name: str):
    """Return an LLMChain (LangChain) using selected HF pipeline."""
    try:
        hf = get_hf_pipeline(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    prompt_template = """Use the provided context to answer the question concisely and clearly.
If the answer is not present in the context, reply exactly: "Answer not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return LLMChain(llm=hf, prompt=prompt)


# ----------------------------
# UI: Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")
    st.markdown(
        """
        **How to use**
        1. Upload one or more PDFs.
        2. Click *Process & Index*. This creates a local FAISS index.
        3. Ask questions in the main panel. Top-k relevant chunks used; source chunks shown below the answer.
        """
    )

    model_choice = st.selectbox(
        "Select local model (text2text)",
        options=["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"],
        index=0,
        help="Larger models give better answers but need more RAM/CPU. Choose accordingly."
    )

    rebuild_index = st.button("Process & Index uploaded PDFs")
    clear_index = st.button("Clear saved FAISS index")
    show_sources = st.checkbox("Show source chunks below answer", value=True)
    top_k = st.slider("Top-k chunks to retrieve", min_value=1, max_value=8, value=5, step=1)

    st.markdown("---")
    st.write("Index status:")
    if os.path.exists(INDEX_DIR) and len(os.listdir(INDEX_DIR)) > 0:
        st.success("Local FAISS index found")
    else:
        st.info("No local index yet")


# ----------------------------
# Main area: Upload + listing
# ----------------------------
uploaded = st.file_uploader("Upload PDF files (you can select multiple)", type=["pdf"], accept_multiple_files=True)

# file preview / names
if uploaded:
    st.markdown("**Uploaded files:**")
    for f in uploaded:
        st.write(f"- {f.name} ({round(len(f.getvalue())/1024,1)} KB)")

# Process & Index (build or append)
if rebuild_index and uploaded:
    with st.spinner("Processing PDFs and building index..."):
        try:
            all_chunks_meta = []
            for f in uploaded:
                file_bytes = f.getvalue()
                raw_text = extract_text_from_pdf_bytes(file_bytes)
                # small cleanup: collapse repeated spaces and trim
                raw_text = " ".join(raw_text.split())
                chunks = split_text_to_chunks(raw_text)
                st.write(f"Extracted {len(chunks)} chunks from {f.name}")
                for i, ch in enumerate(chunks):
                    meta = {"source": f.name, "chunk_id": i}
                    all_chunks_meta.append({"text": ch, "metadata": meta})
            if len(all_chunks_meta) == 0:
                st.error("No text could be extracted from the uploaded PDFs. Check files.")
            else:
                vs = create_or_append_faiss(all_chunks_meta)
                st.success("Vector store created / updated ✅")
        except Exception as e:
            st.error("Failed to create index: " + str(e))
            st.exception(traceback.format_exc())
elif rebuild_index and not uploaded:
    st.warning("No files uploaded. Please upload PDFs then click 'Process & Index'.")

# Clear index
if clear_index:
    try:
        if os.path.exists(INDEX_DIR):
            for fname in os.listdir(INDEX_DIR):
                os.remove(os.path.join(INDEX_DIR, fname))
            st.success("Index cleared.")
        else:
            st.info("No index to clear.")
    except Exception as e:
        st.error("Failed to clear index: " + str(e))


# Load index (if exists)
vector_store = load_vector_store_safe()

# Conversation/session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of (q,a)

# ----------------------------
# Query UI and answering
# ----------------------------
if vector_store is None:
    st.info("No FAISS index available. Upload PDFs and click 'Process & Index' in the sidebar.")
else:
    st.subheader("Ask questions about your PDFs")
    question = st.text_input("Enter your question and press Enter", key="question_input")

    if question:
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            try:
                # retrieve top-k docs (documents are langchain Document instances or dict-like)
                docs = vector_store.similarity_search(question, k=top_k)

                # Normalize docs to (text, metadata)
                normalized = []
                for d in docs:
                    # support both Document and dict
                    text = getattr(d, "page_content", None) or getattr(d, "text", None) or (d.get("text") if isinstance(d, dict) else None)
                    meta = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {})
                    # fallback
                    if text is None:
                        # maybe the stored texts are raw strings
                        text = str(d)
                    normalized.append({"text": text, "metadata": meta or {}})

                # combine top-N chunks into context (limit length to avoid overflow)
                # Join with double newlines to keep chunks distinct
                context = "\n\n".join([c["text"] for c in normalized])

                # Build LLM chain and run
                try:
                    llm_chain = build_llm_chain(model_choice)
                except Exception as e:
                    st.error("Failed to load local model. Try selecting a smaller model in the sidebar. Error: " + str(e))
                    raise

                # Run LLMChain: pass context and question
                response = llm_chain.run({"context": context, "question": question})

                # Show answer and optional sources
                st.markdown("### 💡 Answer")
                st.markdown(response)

                # Save to history
                st.session_state.history.append((question, response))

                if show_sources:
                    st.markdown("### 🔎 Source chunks (top matches)")
                    for i, c in enumerate(normalized):
                        src = c["metadata"].get("source", "unknown")
                        chunk_id = c["metadata"].get("chunk_id", "n/a")
                        st.markdown(f"**{i+1}. Source:** `{src}` — chunk `{chunk_id}`")
                        # show a short excerpt and a copy button
                        excerpt = c["text"]
                        if len(excerpt) > 800:
                            excerpt = excerpt[:800] + " ... (truncated)"
                        st.code(excerpt, language="text")
            except Exception as e:
                st.error("Error during retrieval or generation: " + str(e))
                st.exception(traceback.format_exc())

    # show conversation history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🔁 Conversation history")
        for i, (q, a) in enumerate(reversed(st.session_state.history[-10:])):
            st.write(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
