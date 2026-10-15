import streamlit as st

from lib import db, auth


def render(current_user):
    st.title("Admin — Account Approval")
    st.caption("Approve or reject pending sign-ups, and manage roles for approved users.")

    st.subheader("Pending accounts")
    pending = db.list_pending_users()
    if not pending:
        st.write("No pending accounts.")
    for u in pending:
        with st.container(border=True):
            cols = st.columns([4, 2, 1, 1])
            cols[0].write(f"**{u['username']}**")
            cols[1].write("new sign-up")
            if cols[2].button("Approve", key=f"approve-{u['id']}"):
                db.approve_user(u["id"])
                st.rerun()
            if cols[3].button("Reject", key=f"reject-{u['id']}"):
                db.reject_user(u["id"])
                st.rerun()

    st.subheader("Approved users")
    approved = db.list_approved_users()
    for u in approved:
        with st.container(border=True):
            cols = st.columns([4, 2, 1, 1])
            cols[0].write(f"**{u['username']}**")
            cols[1].write(f"role: {u['role']}")
            is_self = u["username"] == current_user["username"]
            if u["role"] == "admin":
                if cols[2].button("Demote", key=f"demote-{u['id']}", disabled=is_self):
                    db.set_role(u["id"], "user")
                    st.rerun()
            else:
                if cols[2].button("Make admin", key=f"promote-{u['id']}"):
                    db.set_role(u["id"], "admin")
                    st.rerun()
            if cols[3].button("Revoke", key=f"revoke-{u['id']}", disabled=is_self):
                db.revoke_user(u["id"])
                st.rerun()

    if any(u["username"] == current_user["username"] for u in approved):
        st.caption("You can't demote or revoke your own account from here.")
