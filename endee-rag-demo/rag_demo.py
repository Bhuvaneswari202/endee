"""
Endee RAG Demo
==============
Retrieval-Augmented Generation (RAG) pipeline using:
  • Endee        – high-performance vector database
  • sentence-transformers – local text embeddings (all-MiniLM-L6-v2, 384-dim)
  • Google Gemini – LLM for answer generation

Usage:
    python rag_demo.py                        # interactive Q&A loop
    python rag_demo.py --ingest               # (re)ingest documents only
    python rag_demo.py --query "your query"   # single-shot query
"""

import argparse
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

ENDEE_URL: str = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
ENDEE_AUTH_TOKEN: str = os.getenv("ENDEE_AUTH_TOKEN", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INDEX_NAME = "knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 3
GEMINI_MODEL = "gemini-2.0-flash"

console = Console()

# ---------------------------------------------------------------------------
# Sample knowledge base documents
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "id": "doc_01",
        "title": "What is a Vector Database?",
        "text": (
            "A vector database stores data as high-dimensional numerical vectors. "
            "Unlike traditional relational databases that rely on exact keyword matching, "
            "vector databases enable similarity search — finding records that are 'close' "
            "in meaning or context. They are foundational to modern AI applications such as "
            "semantic search, recommendation systems, and Retrieval-Augmented Generation (RAG)."
        ),
        "category": "fundamentals",
    },
    {
        "id": "doc_02",
        "title": "How Endee Works",
        "text": (
            "Endee is a high-performance, open-source vector database written in C++. "
            "It is optimized for modern CPU instruction sets (AVX2, AVX512, NEON, SVE2) "
            "and can handle up to 1 billion vectors on a single node. Endee supports "
            "dense vector retrieval, sparse search for hybrid search, and payload filtering "
            "for metadata-aware queries. It exposes an HTTP API and provides SDKs for "
            "Python, Node.js, Java, and Go."
        ),
        "category": "endee",
    },
    {
        "id": "doc_03",
        "title": "What is RAG (Retrieval-Augmented Generation)?",
        "text": (
            "RAG is an AI architecture that augments a large language model (LLM) with a "
            "retrieval step. Instead of relying solely on the LLM's training data, the system "
            "first retrieves relevant context from an external knowledge base (e.g., a vector "
            "database), then passes that context to the LLM to generate a grounded, accurate "
            "answer. RAG reduces hallucinations and keeps answers factually up to date."
        ),
        "category": "ai",
    },
    {
        "id": "doc_04",
        "title": "Sentence Transformers and Embeddings",
        "text": (
            "Sentence Transformers is a Python library that produces dense vector embeddings "
            "from text using pre-trained transformer models. The model 'all-MiniLM-L6-v2' "
            "produces 384-dimensional embeddings and offers an excellent balance between "
            "speed and accuracy. These embeddings capture the semantic meaning of sentences "
            "and can be compared using cosine similarity to find related passages."
        ),
        "category": "embeddings",
    },
    {
        "id": "doc_05",
        "title": "Cosine Similarity in Vector Search",
        "text": (
            "Cosine similarity measures the angle between two vectors in a high-dimensional "
            "space, returning a value between -1 and 1. A score of 1 means the vectors are "
            "identical in direction (most similar), while 0 means orthogonal (unrelated). "
            "Vector databases like Endee use cosine similarity (or inner product / L2 distance) "
            "to rank results returned by a nearest-neighbour search."
        ),
        "category": "fundamentals",
    },
    {
        "id": "doc_06",
        "title": "Hybrid Search: Dense + Sparse Retrieval",
        "text": (
            "Hybrid search combines dense vector search (semantic similarity) with sparse "
            "vector search (keyword/BM25-style matching). This approach captures both semantic "
            "meaning and exact keyword relevance. Endee supports sparse retrieval alongside "
            "its dense vector capabilities, making it well-suited for production search systems "
            "that need to handle diverse query types."
        ),
        "category": "endee",
    },
    {
        "id": "doc_07",
        "title": "Large Language Models (LLMs)",
        "text": (
            "Large Language Models (LLMs) like Google Gemini, GPT-4, and Claude are deep "
            "neural networks trained on vast text corpora. They excel at natural language "
            "understanding and generation tasks. In RAG systems, the LLM receives a user "
            "query together with retrieved context and generates a coherent, contextually "
            "grounded answer — combining its world knowledge with the provided evidence."
        ),
        "category": "ai",
    },
    {
        "id": "doc_08",
        "title": "Payload Filtering in Endee",
        "text": (
            "Payload filtering allows you to attach arbitrary metadata to stored vectors — "
            "such as category, author, date, or tags — and then filter search results based "
            "on that metadata at query time. For example, you can retrieve only the top-K "
            "vectors that match a specific category or date range, while still ranking them "
            "by semantic similarity. Endee supports payload filtering natively."
        ),
        "category": "endee",
    },
    {
        "id": "doc_09",
        "title": "HNSW Indexing Algorithm",
        "text": (
            "HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest "
            "neighbour (ANN) algorithm widely used in vector databases. It builds a multi-layer "
            "proximity graph over the vectors, enabling very fast approximate nearest-neighbour "
            "lookups with high recall. Endee uses optimized HNSW internals combined with SIMD "
            "CPU instructions to maximize throughput and minimize query latency."
        ),
        "category": "fundamentals",
    },
    {
        "id": "doc_10",
        "title": "Deploying Endee with Docker",
        "text": (
            "Endee can be deployed via Docker on any operating system. The official image "
            "'endeeio/endee-server:latest' is available on Docker Hub. A typical deployment "
            "mounts a local volume for data persistence and exposes port 8080. Optional "
            "token-based authentication can be enabled via the NDD_AUTH_TOKEN environment "
            "variable. Docker Compose is the recommended setup for local development."
        ),
        "category": "deployment",
    },
]


# ---------------------------------------------------------------------------
# Helper: wait for Endee to be ready
# ---------------------------------------------------------------------------
def wait_for_endee(client, retries: int = 10, delay: float = 2.0) -> bool:
    """Poll the Endee server until it responds or retries are exhausted."""
    for attempt in range(1, retries + 1):
        try:
            client.list_indexes()
            return True
        except Exception:
            if attempt < retries:
                console.print(
                    f"  [yellow]Endee not ready yet (attempt {attempt}/{retries}), "
                    f"retrying in {delay}s…[/yellow]"
                )
                time.sleep(delay)
    return False


# ---------------------------------------------------------------------------
# Step 1: Ingest documents into Endee
# ---------------------------------------------------------------------------
def ingest_documents(client, embedder) -> None:
    """Embed all DOCUMENTS and upsert them into the Endee index."""
    from endee import Precision

    console.print("\n[bold cyan]◆ Setting up Endee index…[/bold cyan]")

    # Create or recreate index
    existing_indexes = [idx.name for idx in client.list_indexes()]
    if INDEX_NAME in existing_indexes:
        console.print(f"  Index '[green]{INDEX_NAME}[/green]' already exists — deleting for fresh ingest…")
        client.delete_index(INDEX_NAME)

    client.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        space_type="cosine",
        precision=Precision.INT8,
    )
    console.print(f"  ✓ Created index '[green]{INDEX_NAME}[/green]' (dim={EMBEDDING_DIM}, cosine, INT8)")

    index = client.get_index(name=INDEX_NAME)

    console.print("\n[bold cyan]◆ Embedding and ingesting documents…[/bold cyan]")
    texts = [doc["text"] for doc in DOCUMENTS]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Generating embeddings…", total=None)
        vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        progress.update(task, description="Embeddings ready!")

    # Build upsert payload
    items = []
    for doc, vec in zip(DOCUMENTS, vectors):
        items.append(
            {
                "id": doc["id"],
                "vector": vec.tolist(),
                "meta": {
                    "title": doc["title"],
                    "text": doc["text"],
                    "category": doc["category"],
                },
            }
        )

    index.upsert(items)
    console.print(f"  ✓ Ingested [bold]{len(items)}[/bold] documents into Endee\n")


# ---------------------------------------------------------------------------
# Step 2: Retrieve relevant context from Endee
# ---------------------------------------------------------------------------
def retrieve(client, embedder, query: str) -> list[dict]:
    """Embed the query and fetch the top-K most relevant documents from Endee."""
    index = client.get_index(name=INDEX_NAME)
    query_vector = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    results = index.query(vector=query_vector, top_k=TOP_K)

    docs = []
    for r in results:
        docs.append(
            {
                "id": r.id,
                "similarity": round(r.similarity, 4),
                "title": r.meta.get("title", ""),
                "text": r.meta.get("text", ""),
                "category": r.meta.get("category", ""),
            }
        )
    return docs


# ---------------------------------------------------------------------------
# Step 3: Generate answer with Gemini
# ---------------------------------------------------------------------------
def generate_answer(query: str, context_docs: list[dict]) -> str:
    """Build a RAG prompt from the retrieved context and call Gemini."""
    if not GEMINI_API_KEY:
        return (
            "[bold red]⚠ GEMINI_API_KEY not set.[/bold red] "
            "Please add it to your .env file. "
            "The retrieved context above is shown for reference."
        )

    try:
        from google import genai  # type: ignore

        context_text = "\n\n".join(
            f"[{i+1}] {doc['title']}\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        )

        prompt = f"""You are a helpful AI assistant that answers questions based on provided context.
Use ONLY the information from the context below to answer the question.
If the context does not contain enough information, say so clearly.

---
CONTEXT:
{context_text}
---

QUESTION: {query}

ANSWER:"""

        llm_client = genai.Client(api_key=GEMINI_API_KEY)
        response = llm_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text

    except ImportError:
        return "google-genai is not installed. Run: pip install google-genai"
    except Exception as exc:
        return f"[red]Error calling Gemini API:[/red] {exc}"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def show_retrieved_docs(docs: list[dict]) -> None:
    table = Table(title="Retrieved Documents from Endee", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan", no_wrap=False)
    table.add_column("Category", style="magenta")
    table.add_column("Similarity", justify="right", style="green")
    table.add_column("Snippet", no_wrap=False, max_width=60)

    for i, doc in enumerate(docs, 1):
        snippet = doc["text"][:120] + "…" if len(doc["text"]) > 120 else doc["text"]
        table.add_row(str(i), doc["title"], doc["category"], str(doc["similarity"]), snippet)

    console.print(table)


def run_query(client, embedder, query: str) -> None:
    console.print(Panel(f"[bold white]{query}[/bold white]", title="[yellow]Query[/yellow]", border_style="yellow"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as p:
        t = p.add_task("Searching Endee for relevant context…")
        docs = retrieve(client, embedder, query)
        p.update(t, description="Done!")

    show_retrieved_docs(docs)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as p:
        t = p.add_task("Generating answer with Gemini…")
        answer = generate_answer(query, docs)
        p.update(t, description="Done!")

    console.print(Panel(Markdown(answer), title="[green]Answer[/green]", border_style="green"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Endee RAG Demo")
    parser.add_argument("--ingest", action="store_true", help="(Re)ingest documents into Endee and exit")
    parser.add_argument("--query", type=str, default=None, help="Run a single query and exit")
    args = parser.parse_args()

    # ── Banner ───────────────────────────────────────────────────────────────
    console.print(
        Panel.fit(
            "[bold cyan]Endee RAG Demo[/bold cyan]\n"
            "[dim]Retrieval-Augmented Generation powered by Endee vector database[/dim]",
            border_style="cyan",
        )
    )

    # ── Connect to Endee ─────────────────────────────────────────────────────
    try:
        from endee import Endee
    except ImportError:
        console.print("[red]✗ 'endee' package not found. Run: pip install -r requirements.txt[/red]")
        sys.exit(1)

    auth_token = ENDEE_AUTH_TOKEN or None
    client = Endee(auth_token) if auth_token else Endee()

    # Override base URL if custom ENDEE_URL set
    if ENDEE_URL != "http://localhost:8080/api/v1":
        client.set_base_url(ENDEE_URL)

    console.print(f"\n[dim]Connecting to Endee at {ENDEE_URL}…[/dim]")
    if not wait_for_endee(client):
        console.print(
            "[red]✗ Could not connect to Endee. "
            "Make sure it is running (docker compose up -d) and try again.[/red]"
        )
        sys.exit(1)
    console.print("[green]✓ Connected to Endee[/green]")

    # ── Load embedding model ──────────────────────────────────────────────────
    console.print(f"\n[dim]Loading embedding model '{EMBEDDING_MODEL}'…[/dim]")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        console.print("[red]✗ 'sentence-transformers' not found. Run: pip install -r requirements.txt[/red]")
        sys.exit(1)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    console.print(f"[green]✓ Embedding model ready (dim={EMBEDDING_DIM})[/green]")

    # ── Ingest if needed or explicitly requested ──────────────────────────────
    existing_indexes = [idx.name for idx in client.list_indexes()]
    if args.ingest or INDEX_NAME not in existing_indexes:
        ingest_documents(client, embedder)

    if args.ingest:
        console.print("[bold green]✓ Ingest complete.[/bold green]")
        return

    # ── Single-shot query mode ────────────────────────────────────────────────
    if args.query:
        run_query(client, embedder, args.query)
        return

    # ── Interactive Q&A loop ──────────────────────────────────────────────────
    console.print("\n[bold]Interactive Q&A mode[/bold] — type a question and press Enter. Type [bold red]exit[/bold red] to quit.\n")
    while True:
        try:
            query = console.input("[bold yellow]You:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        run_query(client, embedder, query)
        console.print()


if __name__ == "__main__":
    main()
