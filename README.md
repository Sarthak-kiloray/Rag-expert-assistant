# RAG Expert Assistant

A full Retrieval-Augmented Generation (RAG) system that lets users upload documents and chat with their content in real time.  
Built with **Python**, **LangChain**, **ChromaDB**, and **Gradio**, this project demonstrates the complete end-to-end pipeline of:

- Document ingestion
- Intelligent text chunking
- Embedding generation
- Vector database storage
- Semantic retrieval
- LLM-powered question answering

This is a clean example of a production-style RAG implementation suitable for learning, interviews, and real-world extension.

---

## 🚀 Features

### **1. Multi-file Document Upload**
Supports `.txt` and `.docx` (PDF optional). Uploaded files are parsed, converted to text, chunked, and embedded.

### **2. Intelligent Chunking**
Uses `RecursiveCharacterTextSplitter` to preserve context while creating overlapping chunks for robust retrieval.

### **3. Vector Storage with ChromaDB**
Embeddings are stored locally using a persistent Chroma collection (`vector_db/`).

### **4. OpenAI Embeddings + LLM**
Uses OpenAI’s `text-embedding-3-large` for vectorization and GPT-based models for answering questions.

### **5. Real-time Chat Interface**
Built using **Gradio**, with:
- Live chat history
- Retrieved context display
- Automatic ingestion and pipeline refresh after uploading files

---

## 🧠 Architecture

```text
[Uploaded Files]
        |
        v
[Ingestion / Text Parsing]
        |
        v
[Chunking (Recursive Split)]
        |
        v
[Embedding (OpenAI)]
        |
        v
[ChromaDB Vector Store]
        |
        v
[Retrieval (semantic search)]
        |
        v
[LLM Answer Generation]


