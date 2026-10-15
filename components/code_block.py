import streamlit as st


def code_block(code: str, title: str = None, language: str = "python"):
    if title:
        st.caption(title)
    st.code(code, language=language)
