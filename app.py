import streamlit as st
import os
from PIL import Image


from core.config import setup_app_config
from core.utils import get_device_status, save_uploaded_file
from core.ai_logic import generate_blueprint

env_key = setup_app_config()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")

    # Input de API Key
    api_key_input = st.text_input("OpenAI API Key", type="password", value=env_key)
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.divider()

    # Status do Hardware
    device, msg, status_type = get_device_status()
    if status_type == "success":
        st.success(msg)
    else:
        st.error(msg)

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

    with c1: w = st.number_input("Largura (m)", min_value=0.5, value=3.0, step=0.1, format="%.2f")
    with c2: h = st.number_input("Altura (m)", min_value=0.5, value=2.6, step=0.1, format="%.2f")
    with c3: d = st.number_input("Profundidade (m)", min_value=0.1, value=1.5, step=0.1, format="%.2f")

    prompt_text = st.text_area("O que você deseja criar?", placeholder="Ex: Escritório gamer, luzes neon...",
                               height=100)

    # Botão de Ação
    generate_btn = st.button("🚀 Gerar Prompt Técnico", type="primary", use_container_width=True)

st.divider()

# --- Orquestração do Fluxo ---
if uploaded_file and generate_btn:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("❌ API Key não encontrada.")
    elif not prompt_text:
        st.warning("⚠️ Descreva o que você quer fazer na parede.")
    else:
        col_prev, col_result = st.columns(2)

        # Lado Esquerdo: Imagem Original
        with col_prev:
            st.subheader("Imagem Original")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

            # salva
            save_uploaded_file(uploaded_file)

        # Resultado da IA
        with col_result:
            st.subheader("Processando Lógica...")
            status_box = st.empty()

            try:
                status_box.info("🧠 Consultando Arquiteto AI (GPT-4o-mini)...")

                # Chama Core Logic
                technical_prompt = generate_blueprint(prompt_text, w, h, d, os.environ["OPENAI_API_KEY"])

                status_box.success("✅ Prompt Gerado com Sucesso!")

                st.markdown("### Prompt Técnico (Input para SD):")
                st.code(technical_prompt, language="text")

            except Exception as e:
                status_box.error(f"Erro Crítico: {e}")

elif uploaded_file:
    # Preview Simples
    image = Image.open(uploaded_file)
    st.image(image, caption="Preview", width=400)