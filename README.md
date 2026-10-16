# Custom Generative AI Systems for Enterprises — Course Companion Site

**Group 7** · Goh Kok Boon · Ravikkumar Vignesh Kumar · Yap Wen Lai

An interactive Streamlit companion site for the course: four days of workshop content with
live sandboxes, a built-in AI search assistant with web search and guardrails, and
username/password login with admin approval — all in a single Streamlit app/container.

## Architecture overview

```
                         ┌─────────────────────────────┐
                         │           app.py             │
                         │  (routing + session state)   │
                         └───────────────┬───────────────┘
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             │                           │                           │
      ┌──────▼──────┐           ┌────────▼────────┐          ┌───────▼───────┐
      │  lib/auth.py │           │  lib/db.py       │          │ pages_content/ │
      │  login/      │◄─────────►│  SQLite (users)  │          │ Day1-4, Project│
      │  session     │           │                  │          │ Admin, etc.    │
      └──────────────┘           └──────────────────┘          └────────────────┘
             │
             │ once authenticated + approved
             ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                   components/search_bar.py                        │
   │                  (AI search bar on every page)                    │
   │                                                                     │
   │   1. check_guardrails()  ──────────────►  lib/guardrails.py        │
   │        (unsafe-content filter + topic-scope gate, no LLM call)     │
   │                                                                     │
   │   2. match_site_index()  ──────────────►  lib/site_index.py        │
   │        (keyword-based retrieval over this site's own pages —      │
   │         "RAG without an embedding model")                          │
   │                                                                     │
   │   3. groq_chat_completion()  ──────────►  lib/groq_client.py       │
   │        (Groq-hosted LLM: GPT-OSS 20B/120B, Qwen3)                   │
   │        → may return a tool_call for web_search                     │
   │                                                                     │
   │   4. web_search()  ──────────────────────►  lib/search.py          │
   │        (DuckDuckGo HTML scrape, or Google Custom Search API)       │
   │        results fed back to the LLM as a second grounding pass      │
   └───────────────────────────────────────────────────────────────────┘
```

## Request flow, end to end

1. **`app.py`** boots the app, calls `db.init_db()` (creates the `users` table and seeds the
   admin account from `ADMIN_USERNAME`/`ADMIN_PASSWORD`), then branches on auth state:
   - no session → `pages_content/login.py` (sign in / create account)
   - signed in but not yet approved → `pages_content/pending.py`
   - signed in and approved → sidebar + search bar + the active content page
2. On every authenticated render, the user's row is **re-read from the database**
   (`auth.refresh_current_user()`), so an admin revoking access or changing a role takes
   effect immediately rather than only at next login.
3. The **search bar** is rendered on every authenticated page (`components/search_bar.py`)
   and runs the pipeline below for each query.

## How the LLM is connected

- **Provider:** [Groq](https://groq.com)'s OpenAI-compatible chat completions endpoint
  (`lib/groq_client.py`), called over plain HTTPS with a bearer API key.
- **Models offered:** `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` — picked
  per-query from a dropdown next to the search box.
- **Tool use:** the model is given one tool, `web_search(query)`. When it decides it needs
  current/specific facts, Groq returns a `tool_calls` entry; the app executes the actual
  search (`lib/search.py`) and feeds the results back as a second message, then asks the
  model for a final natural-language answer with tool-calling disabled for that follow-up.
- **Retrieval-augmented grounding:** before the user's message ever reaches the model,
  `lib/site_index.py` scores it against a small hand-written index of this site's own pages
  (title + blurb + keywords per page) and injects the top matches as reference context. This
  is the same retrieve-then-ground shape as a real RAG pipeline — just substring-matching
  instead of embeddings — so the assistant can answer "what's on this site" accurately and
  link the user straight to the right page.
- **Secrets:** `GROQ_API_KEY`, and optionally `GOOGLE_API_KEY`/`GOOGLE_CSE_ID`, are read via
  `lib/config.get_secret()`, which checks OS environment variables first (Docker/`.env`) and
  falls back to Streamlit Community Cloud's `st.secrets` — so the same code runs unmodified
  in both environments.

## Guardrails

Guardrails run **before** any LLM call, so a blocked message never spends API quota and the
decision is never left to the model to police itself (`lib/guardrails.py`):

1. **Unsafe-content filter** — a fixed set of regex patterns (weapons, synthesis of
   controlled substances, CSAM, violence, targeted attacks) that always applies, in every
   mode, with no LLM involved.
2. **Topic-scope gate** — a keyword list covering everything taught across Day 1–4, the
   Group Project, and Model Training Evaluation. In "scoped" modes, a message that doesn't
   match any course keyword is blocked with an explanation instead of being answered.

Four selectable **guardrail modes** (dropdown next to the search box) combine these two
checks differently:

| Mode | Topic-gated? | Web search tool |
|---|---|---|
| Within course | Yes | None — answers from the model's own knowledge |
| DDGS search | Yes | DuckDuckGo (free, keyless) |
| Google search | Yes | Google Custom Search API (needs `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`) |
| Full access | No | DuckDuckGo — general-purpose assistant, demo/debug only |

The unsafe-content filter is never disabled, in any mode.

## Other enhancements

- **Auth:** username/password login, passwords stored as salted PBKDF2 hashes
  (`hashlib.pbkdf2_hmac`, 260k iterations) and compared with `hmac.compare_digest` — never
  plaintext, never `==`. Login runs a dummy hash of equal cost for unknown usernames so
  response timing doesn't leak which usernames exist.
- **Admin approval workflow:** new accounts self-register and start `pending`; an admin
  approves/rejects them, and can promote/demote roles or revoke access for any approved user
  from `pages_content/admin.py`. A seeded admin account (from `ADMIN_USERNAME`/
  `ADMIN_PASSWORD`) guarantees there's always at least one admin on first run.
- **Storage:** SQLite (`data/app.db`), sized for a handful of users — no external database
  service required.
- **Prompt-injection hardening:** web search results are passed to the LLM explicitly labeled
  as untrusted reference data, with an instruction to ignore any embedded commands rather than
  treating a scraped page's text as trusted user input.
- **Deployment:** single `Dockerfile` + `docker-compose.yml` service on port `8501`; also
  deployable directly on Streamlit Community Cloud. `.python-version` pins Python 3.12 so
  dependency installs use prebuilt `numpy`/`pandas` wheels instead of compiling from source.

## Running locally

```bash
cp .env.example .env   # fill in GROQ_API_KEY and admin credentials
docker compose up --build
# → http://localhost:8501
```

Or without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py                   # routing + session state
lib/
  auth.py                # login, session, password hashing
  db.py                  # SQLite persistence (users table)
  config.py              # unified env var / st.secrets reader
  groq_client.py          # Groq chat completions wrapper
  search.py              # DuckDuckGo / Google Custom Search
  guardrails.py          # unsafe-content filter + topic-scope modes
  site_index.py          # keyword-based retrieval over this site's pages
components/
  search_bar.py          # the AI search bar (guardrails → retrieval → LLM → tool use)
  sandboxes.py            # interactive course sandboxes
pages_content/            # one module per page (Day1-4, Project, Admin, Login, Pending, ...)
data/                     # static page content + SQLite database file
```
