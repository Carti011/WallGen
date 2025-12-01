import os
import torch
from dotenv import load_dotenv
import streamlit as st


def load_settings():
    """Carrega variáveis de ambiente e configurações globais."""
    load_dotenv()

    # Configuração visual do Streamlit
    st.set_page_config(
        page_title="WallGen Architect",
        page_icon="🏗️",
        layout="wide"
    )

    # CSS Global
    st.markdown("""
        <style>
            .block-container {padding-top: 1rem; padding-bottom: 0rem;}
            header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    return {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        # M4 exige float32 para o VAE para evitar tela preta
        "torch_dtype": torch.float32,
        "device": "mps" if torch.backends.mps.is_available() else "cpu"
    }