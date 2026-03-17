<div align="center">

# 🔍 Endee RAG Demo

**Retrieval-Augmented Generation (RAG) powered by the [Endee](https://github.com/endee-io/endee) vector database**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Endee](https://img.shields.io/badge/Vector%20DB-Endee-6C63FF)](https://endee.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Required-2496ED?logo=docker)](https://docs.docker.com/get-docker/)

</div>

---

## 📖 Project Overview

This project demonstrates a practical **Retrieval-Augmented Generation (RAG)** pipeline using **Endee** as the vector database. The system:

1. **Embeds** a knowledge base of documents using `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions)
2. **Stores** the resulting vectors with metadata in an Endee index
3. **Retrieves** the most semantically relevant documents for any user query via cosine-similarity search
4. **Generates** a grounded, context-aware answer using the **Google Gemini** LLM

Users can interact with the system via an **interactive terminal Q&A session**, a **single-shot `--query` flag**, or re-ingest the knowledge base at any time.

---

## 🏗️ System Design

```
┌──────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                             │
│                                                                  │
│  ┌─────────────┐   embed    ┌──────────────────────────────────┐ │
│  │  Documents  │ ─────────▶ │        Endee Vector DB           │ │
│  │ (knowledge  │            │  ┌────────────────────────────┐  │ │
│  │   base)     │            │  │  Index: knowledge_base     │  │ │
│  └─────────────┘            │  │  dim=384, cosine, INT8     │  │ │
│                             │  └────────────────────────────┘  │ │
│  ┌─────────────┐   embed    │           ▲  (upsert)            │ │
│  │ User Query  │ ─────────▶ │           │                      │ │
│  └─────────────┘            │  cosine similarity search        │ │
│         │                   │           │                      │ │  
│         │     Top-K docs ◀──┘───────────┘                      │ │
│         ▼                   └──────────────────────────────────┘ │
│  ┌─────────────────────────────────┐                             │
│  │  Prompt = query + retrieved ctx │                             │
│  └─────────────────────────────────┘                             │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                                │
│  │  Gemini LLM  │ ──▶  Grounded Answer                          │
│  └──────────────┘                                                │
└──────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Role |
|---|---|
| **Endee** | Vector database — stores and retrieves embeddings via cosine similarity |
| **sentence-transformers** | Generates 384-dim embeddings locally (no API key needed) |
| **Google Gemini** | LLM that generates the final grounded answer |
| **Docker** | Hosts the Endee server locally |
| **Python** | Orchestrates the pipeline end-to-end |

---

## 🧠 Use of Endee

Endee replaces a traditional keyword search database with **semantic vector similarity search**:

| Feature | How we use it |
|---|---|
| `create_index()` | Creates a `knowledge_base` index with `cosine` space, `INT8` precision, 384 dimensions |
| `index.upsert()` | Stores document embeddings with metadata (title, text, category) |
| `index.query()` | Retrieves the top-K most similar documents for an embedded user query |
| **Payload metadata** | Each vector carries `title`, `text`, and `category` fields accessible in results |
| **Docker deploy** | Endee server runs in a Docker container via `docker-compose.yml` |

**Why Endee?**
- Purpose-built for AI retrieval workloads with high throughput
- Simple HTTP API with a clean Python SDK (`pip install endee`)
- Supports payload metadata filtering for category-aware retrieval
- Runs entirely locally — no external service needed

---

## 🗂️ Project Structure

```
endee-rag-demo/
├── docker-compose.yml    # Starts the Endee vector database server
├── rag_demo.py           # Main RAG pipeline (ingest → retrieve → generate)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore
└── README.md
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Docker Desktop** — [Download](https://docs.docker.com/get-docker/)
- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **Google Gemini API key** — [Get one free](https://aistudio.google.com/app/apikey)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/endee-rag-demo.git
cd endee-rag-demo
```

---

### Step 2 — Configure Environment Variables

Copy the example file and fill in your Gemini API key:

```bash
cp .env.example .env
```

Open `.env` and set:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

### Step 3 — Start Endee (via Docker)

```bash
docker compose up -d
```

This starts the Endee server on **http://localhost:8080**. You can verify it is running by opening that URL in your browser — the Endee dashboard will appear.

---

### Step 4 — Install Python Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

---

### Step 5 — Run the RAG Demo

**Interactive Q&A (recommended):**

```bash
python rag_demo.py
```

The first run will automatically ingest the knowledge base into Endee, then drop you into an interactive question-and-answer loop.

**Single-shot query:**

```bash
python rag_demo.py --query "What is RAG and how does it reduce hallucinations?"
```

**Re-ingest documents only:**

```bash
python rag_demo.py --ingest
```

---

### Example Questions to Try

- *"How does Endee handle vector search?"*
- *"What is the difference between dense and hybrid search?"*
- *"What is cosine similarity and why is it used in vector search?"*
- *"How do I deploy Endee using Docker?"*
- *"What is a vector database?"*

---

## 🛑 Stopping the Server

```bash
docker compose down
```

Your data is persisted in a Docker-managed volume. To wipe everything:

```bash
docker compose down -v
```

---

## 🛠️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key for answer generation |
| `ENDEE_URL` | `http://localhost:8080/api/v1` | Endee server base URL |
| `ENDEE_AUTH_TOKEN` | *(empty)* | Optional auth token if Endee is token-protected |

---

## 📚 References

- [Endee GitHub Repository](https://github.com/endee-io/endee)
- [Endee Documentation](https://docs.endee.io/quick-start)
- [sentence-transformers](https://www.sbert.net/)
- [Google Gemini API](https://ai.google.dev/)

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
