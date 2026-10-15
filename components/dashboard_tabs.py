import streamlit as st


def dashboard_tabs(tabs: list):
    """tabs: list of {label, badge?, render} where render is a zero-arg
    callable. Port of src/components/DashboardTabs.tsx — st.tabs replaces the
    custom tab-strip; badge is folded into the label since st.tabs has no
    separate badge slot."""
    labels = [f"{t['label']} 🔧" if t.get("badge") else t["label"] for t in tabs]
    st_tabs = st.tabs(labels)
    for st_tab, t in zip(st_tabs, tabs):
        with st_tab:
            t["render"]()


def stat_row(stats: list):
    """stats: list of {label, value}. Port of src/components/StatRow.tsx."""
    cols = st.columns(len(stats))
    for col, s in zip(cols, stats):
        with col:
            st.metric(s["label"], s["value"])


def resource_list(resources: list):
    """resources: list of {title, url, desc, headings?, summary?, howTo?}."""
    for r in resources:
        with st.container(border=True):
            st.markdown(f"**[{r['title']}]({r['url']})**")
            st.caption(r.get("desc", ""))
            if r.get("headings"):
                st.write(" · ".join(f"`{h}`" for h in r["headings"]))
            if r.get("summary"):
                st.write(r["summary"])
            if r.get("howTo"):
                st.markdown(f"**How to:** {r['howTo']}")
