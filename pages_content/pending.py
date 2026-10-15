import streamlit as st

from lib import auth


def render(user):
    st.title("Waiting for approval")
    st.info(
        f"Signed in as **{user['username']}**. Your account is pending — one of the admins needs to "
        "approve it before you can see course content. Check back later, or ask an admin directly."
    )
    if st.button("Refresh status"):
        auth.refresh_current_user()
        st.rerun()
    if st.button("Log out"):
        auth.logout()
        st.rerun()
