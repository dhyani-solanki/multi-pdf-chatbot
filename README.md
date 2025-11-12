# 📚 Chat with Multiple PDFs — Local (FAISS + Streamlit + Local HF Models)

A Streamlit app to upload many PDFs, index them with FAISS (using SentenceTransformers embeddings), and chat with their content using local Hugging Face models (e.g., google/flan-t5-*). Ideal for private, offline document Q&A.

---

## 🚀 Features

- 📂 Upload multiple PDF files at once
- 🔎 Extract and clean text using PyPDF2
- 🧠 Create a local FAISS vector store using SentenceTransformer embeddings
- 🤖 Use local Hugging Face text2text pipelines (Flan-T5 family) via `transformers`
- 💬 Ask questions about uploaded PDFs and get concise answers
- 🔗 Show top-k source chunks (filename + chunk id) for transparency
- 💾 Persist FAISS index locally for reuse
- ♻️ Clear / rebuild index from the UI

---

## 🧩 Tech Stack

- **Frontend:** Streamlit  
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)  
- **Vector DB:** FAISS (faiss-cpu)  
- **PDF parsing:** PyPDF2  
- **LLM:** Hugging Face `transformers` pipelines wrapped with LangChain `HuggingFacePipeline` (Flan-T5 recommended)   
- **Storage:** Local filesystem (`faiss_index/`)

---
## Folder Structure
<img width="201" height="189" alt="image" src="https://github.com/user-attachments/assets/1808590c-40e1-4c06-8270-b04fc0d2ac8f" />

