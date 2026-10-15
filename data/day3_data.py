LOADERS_CODE = '''from langchain_community.document_loaders import (
    UnstructuredEPubLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader,
    PyPDFLoader, Docx2txtLoader, JSONLoader, UnstructuredXMLLoader,
    UnstructuredEmailLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter

# Every loader returns a list[Document] with .page_content + .metadata.

pdf_docs = PyPDFLoader("report.pdf").load()
docx_docs = Docx2txtLoader("memo.docx").load()
epub_docs = UnstructuredEPubLoader("book.epub").load()'''

SPLITTER_CODE = '''splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(epub_docs)
# chunk_size=500 keeps each fragment small enough that 5 retrieved chunks
# still fit comfortably inside flan-t5-base's context window.
# chunk_overlap=50 (10% of chunk_size) prevents a sentence that straddles a
# chunk boundary from being split mid-thought.'''

EMBED_CHROMA_CODE = '''from chromadb.utils import embedding_functions
import chromadb
from uuid import uuid4

embed_model_name = "BAAI/bge-small-en-v1.5"
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model_name)

client = chromadb.Client()
collection = client.create_collection(name="christmas_carol", embedding_function=embedder)

collection.add(
    documents=[c.page_content for c in chunks],
    ids=[str(uuid4()) for _ in chunks],
)

results = collection.query(query_texts=["Who is Scrooge?"], n_results=5)'''

RERANK_CODE = '''from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "Who is Scrooge?"
candidates = results["documents"][0]  # top-k from bi-encoder retrieval

pairs = [[query, doc] for doc in candidates]
scores = cross_encoder.predict(pairs)

reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)'''

RAG_QA_CODE = '''from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

question = "Who is Scrooge?"
context = "\\n".join(doc for doc, score in reranked[:5])

prompt = f"Answer the question based on the context.\\n\\nContext: {context}\\n\\nQuestion: {question}"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
outputs = model.generate(**inputs, max_new_tokens=100)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)'''

TWO_STAGE_PIPELINE = [
    {"stage": "Stage 1 — Retrieval (Bi-Encoder)", "desc": "Embeds query & documents independently into vectors, then ranks by cosine similarity. Fast — narrows millions of docs down to top 50-100 candidates."},
    {"stage": "Stage 2 — Re-ranking (Cross-Encoder)", "desc": "Embeds query+document together with full word-by-word cross-attention. Slower but far more accurate."},
]

LOADER_TABLE = [
    {"format": "PDF", "loader": "PyPDFLoader"},
    {"format": "DOCX", "loader": "Docx2txtLoader"},
    {"format": "EPUB", "loader": "UnstructuredEPubLoader"},
    {"format": "PPTX", "loader": "UnstructuredPowerPointLoader"},
    {"format": "XLSX", "loader": "UnstructuredExcelLoader"},
    {"format": "XML", "loader": "UnstructuredXMLLoader"},
    {"format": "Email", "loader": "UnstructuredEmailLoader"},
    {"format": "JSON", "loader": "JSONLoader"},
]

SQL_VS_VECTOR_CODE = '''# SQL (Postgres) — exact-match / structured lookup
SELECT track_id, title, played_at
FROM play_history
WHERE user_id = 42
ORDER BY played_at DESC
LIMIT 20;
# Fast and exact because user_id is indexed (B-tree) — but there is no
# SQL WHERE clause for "tracks that SOUND like this one."

# Vector DB (ChromaDB / pgvector) — similarity lookup
results = collection.query(
    query_embeddings=[now_playing_vector],
    n_results=5,
)
# ChromaDB computes cosine distance via an ANN index like HNSW.

# Real systems use BOTH side by side:
#   Postgres: user accounts, playlists, play history, song metadata
#   Vector DB: the song's audio-feature embedding, for "find similar" queries
# pgvector adds a VECTOR column type + <-> distance operator to Postgres
# so one database can do both if you don't want to run two systems.'''

DATABASE_COMPARISON = [
    {"aspect": "What it stores", "sql": "Rows with fixed columns (user_id, title, artist, play_count) — structured, typed data.",
     "vector": "High-dimensional embedding vectors + metadata."},
    {"aspect": "Query it answers well", "sql": '"Get all songs by artist X" / "Count plays for user 42 this week" — exact filters, joins, aggregates.',
     "vector": '"Find songs that sound like this one" — similarity, not exact match.'},
    {"aspect": "Index type", "sql": "B-tree / hash indexes — exact equality and range lookups.",
     "vector": "ANN (Approximate Nearest Neighbor) indexes like HNSW — fast nearest-neighbor search at scale."},
    {"aspect": "This course's example", "sql": "A conventional Postgres table of users, playlists, and play history.",
     "vector": "ChromaDB storing bge-small-en-v1.5 embeddings of document chunks for retrieval."},
    {"aspect": "Can one DB do both?", "sql": "Postgres alone: no native similarity search.",
     "vector": "pgvector extension adds a VECTOR column + distance operators to Postgres."},
]

DAY3_RESOURCES = [
    {"title": "LangChain document loaders", "url": "https://python.langchain.com/docs/concepts/document_loaders/",
     "desc": "Full catalogue of loaders for PDF/DOCX/EPUB/PPTX/XML/email/JSON.",
     "headings": ["Document loader concept", "Common Document interface"],
     "summary": "Every loader returns a common list[Document] interface, so downstream code never needs format-specific branching.",
     "howTo": "Pick the loader matching your file type, call .load(), and treat the result identically regardless of source."},
    {"title": "RecursiveCharacterTextSplitter", "url": "https://python.langchain.com/docs/concepts/text_splitters/",
     "desc": "Chunking strategy used before embedding.",
     "headings": ["Why split text", "chunk_size / chunk_overlap"],
     "summary": "Splitting keeps each fragment small enough to fit inside an LLM's context window; chunk_overlap prevents boundary-split sentences.",
     "howTo": "RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)."},
    {"title": "BAAI/bge-small-en-v1.5 model card", "url": "https://huggingface.co/BAAI/bge-small-en-v1.5",
     "desc": "Embedding model used to vectorize chunks.",
     "headings": ["Model description", "Evaluation (MTEB benchmark)"],
     "summary": "A compact (33M param) sentence-embedding model producing 384-dimension vectors.",
     "howTo": "SentenceTransformerEmbeddingFunction(model_name=\"BAAI/bge-small-en-v1.5\") plugs into Chroma's embedding_function."},
    {"title": "ChromaDB docs", "url": "https://docs.trychroma.com/",
     "desc": "Vector store used for similarity search.",
     "headings": ["Introduction", "Quickstart"],
     "summary": "Chroma stores embeddings alongside metadata and supports dense/sparse/hybrid search.",
     "howTo": "client.create_collection(...) then collection.add(...) to index, collection.query(...) to retrieve."},
    {"title": "cross-encoder/ms-marco-MiniLM-L-6-v2", "url": "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2",
     "desc": "Cross-encoder used for re-ranking retrieved chunks.",
     "headings": ["Usage", "Performance benchmarks"],
     "summary": "Scores query+passage pairs together with full cross-attention — much more accurate than a bi-encoder.",
     "howTo": "CrossEncoder(\"cross-encoder/ms-marco-MiniLM-L-6-v2\").predict([[query, doc] for doc in candidates])."},
    {"title": "pgvector — vector search in Postgres", "url": "https://github.com/pgvector/pgvector",
     "desc": "Add a VECTOR column + similarity search to a conventional SQL database.",
     "headings": ["Installation", "Querying (distance operators)", "Indexing (HNSW, IVFFlat)"],
     "summary": "pgvector adds a vector column type and distance operators so similarity search runs as plain SQL.",
     "howTo": "CREATE EXTENSION vector; ALTER TABLE songs ADD COLUMN embedding vector(384); then ORDER BY embedding <=> query_vector LIMIT 5."},
    {"title": "Spotify Web API — Audio Features", "url": "https://developer.spotify.com/documentation/web-api",
     "desc": "Real-world precedent for the song-vector sandbox (danceability/energy/valence/acousticness).",
     "headings": ["Get Audio Features for a Track", "Get Recommendations"],
     "summary": "Spotify's API exposes per-track feature vectors — the same embedding-based recommendation pattern this page's sandbox demonstrates at toy scale.",
     "howTo": "Fetch audio features per track, store as vectors in a vector DB, query nearest-neighbors for 'similar songs'."},
]
