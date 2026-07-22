"""
App entry point — defines the two pages and their nav icons, then hands
off to whichever one the user selects.

This replaces the automatic pages/ folder discovery with Streamlit's
explicit st.navigation API, which is what enables clean, professional
Material Symbol icons in the sidebar instead of emoji. (st.set_page_config's
page_icon parameter only controls the browser tab favicon — it doesn't
affect the sidebar nav icon; the sidebar icon comes from st.Page(icon=...)
here, or from an emoji embedded directly in the filename under the old
automatic pages/ convention.)

Run this file, not spread-mech.py directly:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Bloom Energy Tools", layout="wide")

spread_page = st.Page(
    "spread-mech.py",
    title="Spread Calculator",
    icon=":material/monitoring:",
    default=True,
)
options_page = st.Page(
    "pages/1_Options_Calculator.py",
    title="Options Calculator",
    icon=":material/calculate:",
)

pg = st.navigation([spread_page, options_page])
pg.run()