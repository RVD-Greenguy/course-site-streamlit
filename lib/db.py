"""SQLite persistence for users. Passwords are stored as salted hashes only
(see lib/auth.py's hash_password/verify_password) — never in plaintext.
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "app.db")


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL
            )
            """
        )

    _seed_admin()


def _seed_admin():
    """Upsert the admin account named by ADMIN_USERNAME/ADMIN_PASSWORD as an
    approved admin on startup — this is how the first admin gets into the
    system without a manual DB edit. Remaining users/admins are created via
    the login page's 'Create account' tab + admin approval."""
    from .auth import hash_password  # local import avoids a circular import at module load

    username = os.environ.get("ADMIN_USERNAME", "").strip().lower()
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not username or not password:
        return

    now = datetime.now(timezone.utc).isoformat()
    password_hash = hash_password(password)
    with get_conn() as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE users SET password_hash = ?, role = 'admin', status = 'approved' WHERE username = ?",
                (password_hash, username),
            )
        else:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, status, created_at) "
                "VALUES (?, ?, 'admin', 'approved', ?)",
                (username, password_hash, now),
            )


def get_user_by_username(username: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username.lower(),)).fetchone()
        return dict(row) if row else None


def create_pending_user(username: str, password_hash: str):
    """Registers a brand-new account as pending. Raises ValueError if the
    username is already taken."""
    username = username.lower()
    if get_user_by_username(username):
        raise ValueError("That username is already taken.")
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, now),
        )
    return get_user_by_username(username)


def list_pending_users():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE status = 'pending' ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def list_approved_users():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE status = 'approved' ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def approve_user(user_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE users SET status = 'approved' WHERE id = ?", (user_id,))


def reject_user(user_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def set_role(user_id: int, role: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))


def revoke_user(user_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE users SET status = 'pending' WHERE id = ?", (user_id,))
