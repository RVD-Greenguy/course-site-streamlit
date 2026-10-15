import streamlit as st


def evidence(kind: str, source: str, note: str = None):
    """kind: 'course' (lifted from course materials) or 'external' (supplied
    because the course materials skip it) — mirrors src/components/Evidence.tsx."""
    label = "📄 From course materials" if kind == "course" else "⚠ Not in course materials — standard reference"
    icon = st.success if kind == "course" else st.warning
    icon(f"**{label} · {source}**\n\n{note or ''}")
