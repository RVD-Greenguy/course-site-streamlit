"""Reads config from the environment first (Docker/.env via python-dotenv),
falling back to Streamlit Cloud's st.secrets — so the same code works
unmodified whether deployed via docker-compose or Streamlit Community Cloud,
which injects secrets.toml as st.secrets rather than guaranteed OS env vars.
"""
import logging
import os

import streamlit as st

logger = logging.getLogger(__name__)


def get_secret(key: str, default: str = "") -> str:
    value = os.environ.get(key)
    if value:
        return value
    try:
        value = st.secrets.get(key, default)
    except Exception:
        logger.warning("Could not read secret %r from st.secrets (malformed secrets.toml?)", key, exc_info=True)
        return default
    return str(value) if value is not None else default
