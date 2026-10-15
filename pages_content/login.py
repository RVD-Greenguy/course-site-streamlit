import streamlit as st

from lib import auth


def render():
    st.title("Sign in")
    st.caption("Passwords are stored as salted hashes only, never in plaintext.")

    tab_login, tab_register = st.tabs(["Sign in", "Create account"])

    with tab_login:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign in", key="login_submit"):
            if auth.login(username, password):
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_register:
        st.write("New accounts start **pending** — an admin must approve you before you can see course content.")
        new_username = st.text_input("Choose a username", key="register_username")
        new_password = st.text_input("Choose a password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm password", type="password", key="register_confirm")
        if st.button("Create account", key="register_submit"):
            if new_password != confirm_password:
                st.error("Passwords don't match.")
            else:
                try:
                    auth.register_user(new_username, new_password)
                    st.success("Account created — waiting for admin approval. You can sign in now and will see a pending screen.")
                except ValueError as err:
                    st.error(str(err))
