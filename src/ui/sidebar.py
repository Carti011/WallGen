import streamlit as st
import os
import torch

def render_sidebar():
    with st.sidebar:
        st.header("🎛️ Painel de Controle")

        # API Key Manager
        env_key = os.getenv("OPENAI_API_KEY", "")
        user_key = st.text_input("OpenAI API Key", value=env_key, type="password")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key

        st.divider()

        # Hardware Check
        if torch.backends.mps.is_available():
            st.success(f"GPU: M4 (MPS) Ativo ⚡")
        else:
            st.error("GPU: CPU Mode (Lento)")

        st.divider()
        st.info("💡 Dica: O modo Img2Img tenta preservar seu chão e paredes originais.")