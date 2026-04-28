import streamlit as st
from app.ui import render_ui

st.set_page_config(page_title = "AI Fairness Audit", layout = "centered")

render_ui()
