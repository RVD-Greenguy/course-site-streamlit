import streamlit as st


def formula(tex: str, caption: str = None):
    st.latex(tex)
    if caption:
        st.caption(caption)
