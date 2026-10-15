import streamlit as st


def real_world_example(title: str, scenarios: list):
    """scenarios: list of {label, scenario, outcome}. Port of
    src/components/RealWorldExample.tsx — st.tabs replaces the clickable
    pill-row, st.info replaces the highlighted outcome line."""
    with st.container(border=True):
        st.markdown(f"**🌍 Real-world example — {title}**")
        if len(scenarios) == 1:
            s = scenarios[0]
            st.write(s["scenario"])
            st.info(f"→ {s['outcome']}")
        else:
            tabs = st.tabs([s["label"] for s in scenarios])
            for tab, s in zip(tabs, scenarios):
                with tab:
                    st.write(s["scenario"])
                    st.info(f"→ {s['outcome']}")
