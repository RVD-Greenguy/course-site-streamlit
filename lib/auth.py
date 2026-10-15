"""Username/password auth. Passwords are stored as salted PBKDF2 hashes
(hashlib.pbkdf2_hmac, stdlib only — no extra dependency) — never in
plaintext, and never compared with a plain '==' (constant-time compare).
"""
import hashlib
import hmac
import os

import streamlit as st

from . import db

SESSION_KEY = "auth_user"
_PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"{salt.hex()}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, derived_hex = stored_hash.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(derived_hex)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return hmac.compare_digest(actual, expected)


def register_user(username: str, password: str):
    """Creates a new pending account. Raises ValueError if the username is
    taken or the inputs are invalid."""
    username = username.strip().lower()
    if not username or not password:
        raise ValueError("Username and password are required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    return db.create_pending_user(username, hash_password(password))


_DUMMY_HASH = hash_password(os.urandom(16).hex())


def login(username: str, password: str) -> bool:
    user = db.get_user_by_username(username.strip().lower())
    stored_hash = user["password_hash"] if user else _DUMMY_HASH
    if not verify_password(password, stored_hash) or not user:
        return False
    st.session_state[SESSION_KEY] = user
    return True


def current_user():
    return st.session_state.get(SESSION_KEY)


def refresh_current_user():
    """Re-read the logged-in user's row from the DB — call after an admin
    action might have changed this user's own status/role mid-session."""
    user = current_user()
    if not user:
        return None
    fresh = db.get_user_by_username(user["username"])
    if fresh:
        st.session_state[SESSION_KEY] = fresh
    return fresh


def logout():
    st.session_state.pop(SESSION_KEY, None)


def is_admin(user) -> bool:
    return bool(user) and user.get("role") == "admin"


def is_approved(user) -> bool:
    return bool(user) and user.get("status") == "approved"
