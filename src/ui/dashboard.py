import streamlit as st
import os

from src.utils.file_manager import save_uploaded_image, save_generated_image
from src.services.llm_architect import create_technical_prompt
from src.services.image_renderer import render_room


def render_dashboard():
    st.title("WallGen Architect 🏗️")
    st.markdown("### Virtual Staging & Decoração com IA")

    # Layout Principal
    col_input, col_settings = st.columns([1, 1])

    uploaded_file = None
    with col_input:
        uploaded_file = st.file_uploader("📸 Foto do Ambiente", type=["jpg", "png"])

    with col_settings:
        c1, c2, c3 = st.columns(3)
        w = c1.number_input("Largura (m)", 1.0, 10.0, 3.0)
        h = c2.number_input("Altura (m)", 1.0, 5.0, 2.8)
        d = c3.number_input("Profundidade (m)", 1.0, 10.0, 2.0)

        user_request = st.text_area("🛋️ O que você quer adicionar?",
                                    placeholder="Ex: Uma mesa de escritório de madeira com um iMac...",
                                    height=100)

        btn_generate = st.button("✨ Projetar Novo Ambiente", type="primary", use_container_width=True)

    st.divider()

    # Lógica de Orquestração
    if uploaded_file and btn_generate:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("⚠️ Configure a API Key na barra lateral.")
            return

        # Salva
        img_path = save_uploaded_image(uploaded_file)

        col_orig, col_res = st.columns(2)

        with col_orig:
            st.subheader("Original")
            st.image(img_path, use_container_width=True)

        with col_res:
            st.subheader("Proposta da IA")
            status = st.status("👷 Trabalhando no projeto...", expanded=True)

            try:
                # Gerar Prompt
                status.write("🧠 Planejando arquitetura...")
                tech_prompt = create_technical_prompt(user_request, w, h, d, os.environ["OPENAI_API_KEY"])
                st.caption(f"Prompt Técnico: {tech_prompt}")

                # Renderiza
                status.write("🎨 Renderizando (Isso preserva as paredes)...")
                new_img, edges = render_room(tech_prompt, img_path)

                # Salva
                save_path = save_generated_image(new_img)

                status.update(label="Concluído!", state="complete", expanded=False)

                st.image(new_img, caption=f"Salvo em: {save_path}", use_container_width=True)
                with st.expander("Ver Estrutura"):
                    st.image(edges, use_container_width=True)

            except Exception as e:
                status.update(label="Erro!", state="error")
                st.error(f"Falha no processo: {e}")

    elif uploaded_file:
        st.image(uploaded_file, width=300, caption="Preview")