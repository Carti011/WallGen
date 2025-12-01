import streamlit as st
import os
from PIL import Image

# Imports da Arquitetura
from core.config import setup_app_config
from core.utils import get_device_status, save_uploaded_file
from core.ai_logic import generate_blueprint
from core.image_gen import generate_image

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

    st.info("💡 Modo M4: Rodando em Alta Precisão (FP32) para evitar erros gráficos.")

# --- Interface Principal ---
st.title("WallGen 🧱 <MVP>")

col_upload, col_params = st.columns([1, 1])
uploaded_file = None

with col_upload:
    st.info(" Upload da Parede")
    uploaded_file = st.file_uploader("Imagem base", type=["jpg", "jpeg", "png"])

with col_params:
    st.info(" Dimensões & Pedido")
    c1, c2, c3 = st.columns(3)

    with c1: w = st.number_input("Largura (m)", min_value=0.5, value=3.0, step=0.1, format="%.2f")
    with c2: h = st.number_input("Altura (m)", min_value=0.5, value=2.6, step=0.1, format="%.2f")
    with c3: d = st.number_input("Profundidade (m)", min_value=0.1, value=1.5, step=0.1, format="%.2f")

    prompt_text = st.text_area("O que você deseja criar?", placeholder="Ex: Escritório gamer, luzes neon...",
                               height=100)

    generate_prompt_btn = st.button("🚀 Gerar Prompt Técnico", type="primary", use_container_width=True)

st.divider()

# Variáveis de Sessão
if "technical_prompt" not in st.session_state:
    st.session_state.technical_prompt = None
if "input_path" not in st.session_state:
    st.session_state.input_path = None

#  Gerar Texto (GPT)
if uploaded_file and generate_prompt_btn:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("❌ API Key não encontrada.")
    elif not prompt_text:
        st.warning("⚠️ Descreva o pedido.")
    else:
        save_path = save_uploaded_file(uploaded_file)
        st.session_state.input_path = save_path

        try:
            with st.spinner("🧠 Arquiteto pensando..."):
                prompt = generate_blueprint(prompt_text, w, h, d, os.environ["OPENAI_API_KEY"])
                st.session_state.technical_prompt = prompt
                st.success("Prompt Criado!")
        except Exception as e:
            st.error(f"Erro GPT: {e}")

# Visualização e Geração (Stable Diffusion)
if st.session_state.input_path:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Original")
        st.image(st.session_state.input_path, use_column_width=True)

    with col_r:
        if st.session_state.technical_prompt:
            st.subheader("Prompt Técnico")
            st.info(st.session_state.technical_prompt)

            if st.button("🎨 Renderizar Imagem (Local M4)", type="primary"):
                try:
                    with st.spinner("🎨 Pintando pixels (FP32)..."):
                        generated_img, edge_map = generate_image(
                            st.session_state.technical_prompt,
                            st.session_state.input_path
                        )

                        st.subheader("Resultado")
                        st.image(generated_img, caption="Render Final", use_column_width=True)

                        with st.expander("Ver o que a IA 'enxergou' (Estrutura)"):
                            st.image(edge_map, caption="ControlNet Canny Map", use_column_width=True)

                except Exception as e:
                    st.error(f"Erro na Geração: {e}")
        else:
            st.write("Gere o prompt técnico primeiro.")