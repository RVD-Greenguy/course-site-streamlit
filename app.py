import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from lib import db, auth
from components.search_bar import render_search_bar
from pages_content import (
    login, pending, admin, overview, day1, day2, day3, day4,
    project, model_evaluation, cheatsheet, resources,
)

st.set_page_config(page_title="Custom Generative AI Systems for Enterprises", layout="wide")
db.init_db()

PAGES = [
    "Overview", "Day 1", "Day 2", "Day 3", "Day 4",
    "Group Project", "Model Training Evaluation", "Cheatsheet", "Resources",
]


def navigate(page_name: str):
    st.session_state.active_page = page_name
    st.rerun()


def render_sidebar(user):
    with st.sidebar:
        st.markdown("**Custom Generative AI**")
        st.caption("Systems for Enterprises · NUS-ISS")
        st.caption("**Group 7**\n\nGoh Kok Boon\n\nRavikkumar Vignesh Kumar\n\nYap Wen Lai")
        st.divider()

        pages = list(PAGES)
        if auth.is_admin(user):
            pages.append("Admin")

        active = st.session_state.get("active_page", "Overview")
        if active not in pages:
            active = "Overview"
        choice = st.radio("Navigate", pages, index=pages.index(active), label_visibility="collapsed")
        if choice != active:
            st.session_state.active_page = choice
            st.rerun()

        st.divider()
        st.caption(f"Signed in as **{user['username']}**")
        if st.button("Log out"):
            auth.logout()
            st.rerun()


def main():
    user = auth.current_user()

    if not user:
        login.render()
        return

    if not auth.is_approved(user):
        pending.render(user)
        return

    render_sidebar(user)
    render_search_bar(page_registry=None)

    active = st.session_state.get("active_page", "Overview")
    if active == "Overview":
        overview.render(navigate)
    elif active == "Day 1":
        day1.render()
    elif active == "Day 2":
        day2.render()
    elif active == "Day 3":
        day3.render()
    elif active == "Day 4":
        day4.render()
    elif active == "Group Project":
        project.render()
    elif active == "Model Training Evaluation":
        model_evaluation.render(navigate)
    elif active == "Cheatsheet":
        cheatsheet.render()
    elif active == "Resources":
        resources.render()
    elif active == "Admin" and auth.is_admin(user):
        admin.render(user)
    else:
        overview.render(navigate)


if __name__ == "__main__":
    main()
