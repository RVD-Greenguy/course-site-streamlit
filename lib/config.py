"""Reads config from the environment first (Docker/.env via python-dotenv),
falling back to Streamlit Cloud's st.secrets — so the same code works
unmodified whether deployed via docker-compose or Streamlit Community Cloud,
which injects secrets.toml as st.secrets rather than guaranteed OS env vars.
"""
import os

import streamlit as st


def get_secret(key: str, default: str = "") -> str:
    value = os.environ.get(key)
    if value:
        return value
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default
