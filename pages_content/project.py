import streamlit as st

FEATURES = [
    {"area": "Day 1 — Summarization & Multimodal", "detail": "Sandboxes: Summarization, Chat Template, Sampling Math, Perplexity, Token Cost, Float Precision, ROUGE/BLEU calculators."},
    {"area": "Day 2 — LoRA Fine-tuning", "detail": "Sandboxes: LoRA adapter math, Scaling Law explorer, Embedding Similarity calculator."},
    {"area": "Day 3 — Document Loaders & RAG", "detail": "Sandboxes: Chunking strategy, RAG pipeline walk-through, Rerank comparison, SQL-vs-vector-DB song recommender."},
    {"area": "Day 4 — Pre-training & Agentic AI", "detail": "Sandboxes: Pre-train loss curves, CartPole (Q-learning), Tool-use loop, Tool building, Routing, Workflow pattern."},
]

ARCHITECTURE = [
    {"name": "Frontend/Backend", "detail": "Single Streamlit app/container — no separate nginx/Express layer; Python renders every page and calls Groq/search directly in-process."},
    {"name": "AI search bar", "detail": "Top search bar wired to a Groq-hosted open model (GPT-OSS 20B/120B, Qwen3). The model has a web_search tool it can call for current-events questions."},
    {"name": "Guardrails", "detail": "Modular — a dropdown next to the search bar picks between 4 modes (Within course, DDGS search, Google search, Full access), each independently controlling the topic scope and which search engine the model can call."},
    {"name": "Auth", "detail": "Username/password login — passwords stored as salted PBKDF2 hashes, never plaintext. New accounts self-register and start pending; admins approve them from the Admin page."},
    {"name": "Storage", "detail": "SQLite (users table) — sized for a handful of users and admins, no external database service needed."},
    {"name": "Deployment", "detail": "Single Dockerfile + docker-compose service on :8501."},
]


def render():
    st.title("Group Project — Group 7")
    st.caption(
        "This website *is* the group project: an interactive companion site for the course, covering "
        "all four workshop days with live sandboxes, a built-in AI search assistant with web search "
        "and guardrails, and username/password login with admin approval — all in one Streamlit app."
    )
    st.caption("**Members:** Goh Kok Boon · Ravikkumar Vignesh Kumar · Yap Wen Lai")

    with st.container(border=True):
        st.subheader("What we built")
        st.write(
            "Instead of a one-off notebook, Group 7 built a deployable product: a course-notes site "
            "where every workshop concept gets its own runnable sandbox, backed by passwordless "
            "login with admin-gated access and its own AI assistant — not a static write-up."
        )

    st.header("Architecture")
    st.table(ARCHITECTURE)

    st.header("Course content, as working sandboxes")
    st.table(FEATURES)

    st.header("Login & admin approval")
    st.write(
        "Sign in with a username and password — passwords are stored as salted hashes, never "
        "plaintext. New users create their own account from the login page's \"Create account\" "
        "tab; every new account starts **pending** until an admin approves it from the Admin page. "
        "A seed admin account (`group007`) is provisioned on first run via environment variables."
    )

    st.header("The AI search bar")
    st.write(
        "A second dropdown next to the model picker chooses the guardrail mode: Within course "
        "(topic-gated, no web search), DDGS search or Google search (topic-gated plus a web_search "
        "tool), or Full access (no topic restriction). A separate, always-on unsafe-content filter "
        "blocks dangerous queries in every mode."
    )

    st.header("What to show in the demo")
    st.write(
        "- Walk through the sidebar: each day's page and at least one sandbox per day actually running.\n"
        "- Create a new account and show the pending screen, then approve it from Admin.\n"
        "- Ask the search bar a question that should get blocked — show the guardrail firing.\n"
        "- `docker compose up --build` from a clean checkout, showing the single container come up on port 8501."
    )
