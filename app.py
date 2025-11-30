import streamlit as st
import os
from PIL import Image
import torch
from dotenv import load_dotenv

load_dotenv()

# --- Configuração da Página ---
st.set_page_config(
    page_title="WallGen MVP",
    page_icon="🧱",
    layout="wide"
)

# --- CSS para layout compacto ---
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: Configurações ---
with st.sidebar:
    st.header("⚙️ Configurações")

    # Recupera a chave do ambiente ou deixa vazio
    env_key = os.getenv("OPENAI_API_KEY", "")

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=env_key,
        help="Lida automaticamente do arquivo .env se configurado."
    )

    # Atualiza a variável de ambiente na sessão se o usuário digitou algo novo
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.divider()

    # Validação do M4 (MPS)
    st.subheader("🖥️ Status do Hardware")
    if torch.backends.mps.is_available():
        st.success("Apple Silicon (MPS): ATIVO 🚀")
        device = "mps"
    else:
        st.error("Apple Silicon (MPS): INATIVO (CPU) ⚠️")
        device = "cpu"

# --- Interface Principal ---
st.title("WallGen 🧱 <MVP>")

# Layout: Upload e Inputs
col_upload, col_params = st.columns([1, 1])
uploaded_file = None

with col_upload:
    st.info("1. Upload da Parede")
    uploaded_file = st.file_uploader("Imagem base", type=["jpg", "jpeg", "png"])

with col_params:
    st.info("2. Dimensões Reais")
    c1, c2 = st.columns(2)
    with c1:
        width = st.number_input("Largura (m)", min_value=1.0, value=3.0, step=0.1)
    with c2:
        height = st.number_input("Altura (m)", min_value=1.0, value=2.6, step=0.1)

    st.caption(f"Área: {width * height:.2f} m²")

st.divider()

# Preview
if uploaded_file:
    try:
        image = Image.open(uploaded_file)

        # Salva temporariamente
        os.makedirs("temp_data/uploads", exist_ok=True)
        save_path = os.path.join("temp_data/uploads", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.subheader("Visualização")
        st.image(image, caption="Imagem Base", use_column_width=True)

    except Exception as e:
        st.error(f"Erro: {e}")
else:
    st.write("Aguardando imagem...")

# Rodapé de Debug
st.markdown("---")
# Verifica se a chave foi carregada (imprime apenas os primeiros caracteres por segurança)
key_status = "Carregada ✅" if os.environ.get("OPENAI_API_KEY") else "Ausente ❌"
st.caption(f"System: {device.upper()} | API Key: {key_status}")