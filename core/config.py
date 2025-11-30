import streamlit as st
import os
from dotenv import load_dotenv


def setup_app_config():
    """Configurações iniciais da página, CSS e Variáveis de Ambiente."""

    load_dotenv()

    st.set_page_config(
        page_title="WallGen MVP",
        page_icon="🧱",
        layout="wide"
    )

    st.markdown("""
        <style>
            .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        </style>
    """, unsafe_allow_html=True)

    return os.getenv("OPENAI_API_KEY", "")