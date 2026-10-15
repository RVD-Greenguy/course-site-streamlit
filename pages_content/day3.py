import streamlit as st

from components.code_block import code_block
from components.evidence import evidence
from components.real_world_example import real_world_example
from components.dashboard_tabs import dashboard_tabs, stat_row, resource_list
from components.sandboxes import (
    chunking_sandbox, embedding_similarity_calculator, rerank_sandbox,
    rag_pipeline_sandbox, music_vector_db_sandbox,
)
from data.day3_data import (
    LOADERS_CODE, SPLITTER_CODE, EMBED_CHROMA_CODE, RERANK_CODE, RAG_QA_CODE,
    TWO_STAGE_PIPELINE, LOADER_TABLE, DAY3_RESOURCES, SQL_VS_VECTOR_CODE, DATABASE_COMPARISON,
)


def render():
    st.title("Day 3 — Document Loaders & RAG")
    st.caption('Loading and chunking documents, building a vector store, and a full retrieval-augmented Q&A pipeline over "A Christmas Carol".')

    evidence("external", "Gap in course materials",
             "Both Day 3 workshop notebooks are entirely unfilled TODO scaffolds — everything below is a complete reference implementation using the exact model names the blank notebook names.")

    stat_row([
        {"label": "Chunk size", "value": "~500 chars"},
        {"label": "Embedding dims", "value": "384 (bge-small)"},
        {"label": "Retrieve → rerank", "value": "50 → 5"},
        {"label": "Answer model", "value": "flan-t5-base"},
    ])

    st.header("Pipeline dashboard")

    def _pipeline():
        rag_pipeline_sandbox()
        real_world_example("RAG replacing a brittle search box", [
            {"label": "Internal wiki search", "scenario": "An engineering team's Confluence has 3,000 pages; keyword search returns 40 loosely-related pages.",
             "outcome": "RAG embeds the question and retrieves by meaning, not exact words, then synthesizes a direct answer with citations."},
            {"label": "Customer-facing docs bot", "scenario": "A SaaS docs site gets the same question in 50 different phrasings a day.",
             "outcome": "One RAG pipeline handles all phrasings — the embedding model maps semantically similar questions near each other."},
        ])

    def _loaders():
        st.table(LOADER_TABLE)
        code_block(LOADERS_CODE, "document loaders")
        real_world_example("messy real-world document sets", [
            {"label": "Legal/compliance", "scenario": "A law firm's knowledge base mixes scanned PDFs, Word contracts, and emailed PDF amendments.",
             "outcome": "Multiple loaders normalize to the same Document interface — one pipeline handles all formats."},
            {"label": "Finance reporting", "scenario": "A finance team's RAG corpus is PowerPoint decks and Excel spreadsheets.",
             "outcome": "Format-specific loaders extract into the same chunkable form — downstream code never changes."},
        ])

    def _chunking():
        code_block(SPLITTER_CODE, "RecursiveCharacterTextSplitter")
        chunking_sandbox()
        real_world_example("chunk size is a real design decision", [
            {"label": "Too small", "scenario": "A policy-FAQ bot chunks a legal clause into 50-character pieces.",
             "outcome": "A clause gets split mid-sentence, losing a critical exception — wrong answer, confidently given."},
            {"label": "Too large", "scenario": "The same bot chunks a whole 10-page document as one chunk.",
             "outcome": "The embedding averages over 10 pages of unrelated topics — retrieval quality drops."},
        ])

    def _embeddings():
        code_block(EMBED_CHROMA_CODE, "BAAI/bge-small-en-v1.5 + Chroma")
        embedding_similarity_calculator()
        real_world_example("embeddings beyond text search", [
            {"label": "Duplicate ticket detection", "scenario": "A support team gets the same bug reported 30 different ways.",
             "outcome": "Cosine similarity above ~0.85 auto-suggests a merge, with no keyword matching."},
            {"label": "Content recommendation", "scenario": "A news app wants 'articles similar to this one.'",
             "outcome": "Embed once; recommend nearest neighbors by cosine similarity — same math as RAG retrieval."},
        ])

    def _rerank():
        st.table(TWO_STAGE_PIPELINE)
        code_block(RERANK_CODE, "cross-encoder/ms-marco-MiniLM-L-6-v2")
        rerank_sandbox()
        real_world_example("why bother reranking at all", [
            {"label": "E-commerce search", "scenario": 'A shopper searches "waterproof jacket for toddler" — bi-encoder top-50 includes adult jackets and umbrellas.',
             "outcome": "The cross-encoder reads query+candidate together and ranks the actual best match higher."},
        ])

    def _databases():
        st.write(
            "Day 3's slides never say the word 'database' out loud, but ChromaDB **is** a database — "
            "just a specialized one. Real apps run a vector database (similarity search) alongside a "
            "conventional SQL database (everything else: users, orders, play counts)."
        )
        st.table(DATABASE_COMPARISON)
        code_block(SQL_VS_VECTOR_CODE, "exact lookup (SQL) vs similarity lookup (vector DB)")
        music_vector_db_sandbox()

    def _qa():
        code_block(RAG_QA_CODE, "FLAN-based question answering over retrieved context")
        real_world_example("grounded answers vs hallucination", [
            {"label": "Without RAG", "scenario": "A base LLM is asked about a company's refund policy with no company data in training.",
             "outcome": "The model confidently invents a plausible-sounding but wrong policy — hallucination."},
            {"label": "With RAG", "scenario": "The same question, but retrieved chunks are injected into the prompt.",
             "outcome": "The model answers from the provided text — wrong answers now mean a debuggable retrieval failure, not an ungrounded hallucination."},
        ])

    dashboard_tabs([
        {"label": "Full pipeline", "render": _pipeline},
        {"label": "Loaders", "render": _loaders},
        {"label": "Chunking", "badge": True, "render": _chunking},
        {"label": "Embeddings", "badge": True, "render": _embeddings},
        {"label": "Retrieve + rerank", "badge": True, "render": _rerank},
        {"label": "Databases", "badge": True, "render": _databases},
        {"label": "RAG Q&A", "render": _qa},
    ])

    st.header("Resources")
    resource_list(DAY3_RESOURCES)
