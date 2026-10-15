import streamlit as st

from components.dashboard_tabs import resource_list
from data.day1_data import DAY1_RESOURCES
from data.day2_data import DAY2_RESOURCES
from data.day3_data import DAY3_RESOURCES
from data.day4_data import DAY4_RESOURCES


def render():
    st.title("Resources")
    st.caption("Curated external links, organized by day.")

    groups = [
        ("Day 1 — Summarization & Multimodal", DAY1_RESOURCES),
        ("Day 2 — Fine-tuning (LoRA)", DAY2_RESOURCES),
        ("Day 3 — Document Loaders & RAG", DAY3_RESOURCES),
        ("Day 4 — Pre-training & Agentic AI", DAY4_RESOURCES),
    ]
    for title, items in groups:
        st.header(title)
        resource_list(items)
