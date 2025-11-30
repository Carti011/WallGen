import streamlit as st
import os
from PIL import Image
import torch
from dotenv import load_dotenv

# Importa a função da pasta 'core'
from core.ai_logic import generate_blueprint

load_dotenv()

# --- Configuração da Página ---
st.set_page_config(page_title="WallGen MVP", page_icon="🧱", layout="wide")
st.markdown("""<style>.block-container {padding-top: 1rem; padding-bottom: 0rem;}</style>""", unsafe_allow_html=True)

# --- Sidebar: Configurações ---
with st.sidebar:
    st.header("⚙️ Configurações")

    # Gestão da API Key
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.text_input("OpenAI API Key", type="password", value=env_key)
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.divider()

    # Monitor de Hardware (M4)
    if torch.backends.mps.is_available():
        st.success("Apple Silicon (MPS): ATIVO 🚀")
    else:
        st.error("MPS INATIVO (CPU) ⚠️")

# --- Interface Principal ---
st.title("WallGen 🧱 <MVP>")

col_upload, col_params = st.columns([1, 1])
uploaded_file = None

with col_upload:
    st.info("1. Upload da Parede")
    uploaded_file = st.file_uploader("Imagem base", type=["jpg", "jpeg", "png"])

with col_params:
    st.info("2. Dimensões & Pedido")
    c1, c2, c3 = st.columns(3)

    with c1:
        # Largura: Padrão 3m, Mínimo 0.5m
        w = st.number_input("Largura (m)", min_value=0.5, value=3.0, step=0.1, format="%.2f")
    with c2:
        # Altura: Padrão 2.6m
        h = st.number_input("Altura (m)", min_value=0.5, value=2.6, step=0.1, format="%.2f")
    with c3:
        # Profundidade: Padrão 1.5m
        d = st.number_input("Profundidade (m)", min_value=0.1, value=1.5, step=0.1, format="%.2f")

    prompt_text = st.text_area("O que você deseja criar?", placeholder="Ex: Escritório gamer minimalista...",
                               height=100)

    # Botão de Ação
    generate_btn = st.button("🚀 Gerar Prompt Técnico", type="primary", use_container_width=True)

st.divider()

# --- Lógica de Execução ---
if uploaded_file and generate_btn:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("❌ API Key não encontrada.")
    elif not prompt_text:
        st.warning("⚠️ Descreva o que você quer fazer na parede.")
    else:
        col_prev, col_result = st.columns(2)

        with col_prev:
            st.subheader("Imagem Original")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

            # Salva temporariamente para uso futuro
            os.makedirs("temp_data/uploads", exist_ok=True)
            image.save(os.path.join("temp_data/uploads", "input.jpg"))

        with col_result:
            st.subheader("Processando Lógica...")
            status_box = st.empty()

            try:
                status_box.info("🧠 Consultando Arquiteto AI (GPT-4o-mini)...")

                # Chamada ao Backend (Core)
                technical_prompt = generate_blueprint(prompt_text, w, h, d, os.environ["OPENAI_API_KEY"])

                status_box.success("✅ Prompt Gerado com Sucesso!")

                st.markdown("### Prompt Técnico (Input para SD):")
                st.code(technical_prompt, language="text")

            except Exception as e:
                status_box.error(f"Erro Crítico: {e}")

elif uploaded_file:
    # Preview apenas visual
    image = Image.open(uploaded_file)
    st.image(image, caption="Preview", width=400)