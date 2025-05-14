import streamlit as st

st.set_page_config(
    page_title="Leitor de Controle de Cotações - IPC",
    page_icon="../assets/logo_fgv.png"
)

st.logo("../assets/logo_ibre.png")

SHEET_NAMES = ["SP", "RS", "RJ", "PE", "MG", "DF", "BA"]
