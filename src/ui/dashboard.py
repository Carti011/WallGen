import streamlit as st
import os

from src.utils.file_manager import save_uploaded_image, save_generated_image
from src.services.llm_architect import create_technical_prompt
from src.services.image_renderer import render_room


def render_dashboard():
    st.markdown("## 🏗️ WallGen Architect <span style='font-size:0.6em; color:gray'>MVP M4 Edition</span>",
                unsafe_allow_html=True)

    # Layout Principal
    with st.container():
        col_left, col_right = st.columns([1, 1.2], gap="large")

        with col_left:
            st.info("Passo 1: Fotografe o ambiente e defina os limites.")
            uploaded_file = st.file_uploader("📸 Upload do Ambiente (Vazio ou Parede)", type=["jpg", "png"])

            with st.expander("📐 Dimensões & Restrições", expanded=True):
                c1, c2 = st.columns(2)
                w = c1.number_input("Largura Parede (m)", 1.0, 15.0, 3.0, step=0.1)
                h = c2.number_input("Pé Direito (m)", 2.0, 6.0, 2.7, step=0.1)

                st.markdown("---")
                st.caption("🔴 Crítico: Qual a profundidade máxima para os móveis não invadirem o corredor?")
                d = st.slider("Profundidade Disponível (m)", 0.3, 3.0, 1.2, step=0.1,
                              help="Isso define se o móvel será 'Slim' ou 'Profundo'.")

            st.write("🛋️ **Briefing do Projeto**")
            user_request = st.text_area("O que você deseja criar?",
                                        placeholder="Ex: Quero transformar em um Home Office minimalista com madeira clara...",
                                        height=120)

            with st.expander("⚙️ Ajuste Fino da IA"):
                guidance = st.slider("Fidelidade ao Prompt", 5.0, 15.0, 8.5)
                strength = st.slider("Interferência na Arquitetura", 0.3, 0.9, 0.65,
                                     help="Quanto maior, mais a IA muda a estrutura original da parede.")

            generate_btn = st.button("✨ Renderizar Projeto", type="primary", use_container_width=True)

        with col_right:
            if uploaded_file:
                # Salva
                img_path = save_uploaded_image(uploaded_file)

                if not generate_btn:
                    st.image(img_path, caption="Ambiente Original", use_container_width=True)

                else:
                    if not os.environ.get("OPENAI_API_KEY"):
                        st.error("⚠️ API Key não configurada no menu lateral.")
                        st.stop()

                    tabs = st.tabs(["🎨 Render", "🏗️ Estrutura (Debug)"])

                    with tabs[0]:
                        status_container = st.status("👷 Iniciando Obras Digitais...", expanded=True)

                        try:
                            # Inteligência Arquitetônica
                            status_container.write("🧠 Calculando restrições físicas e estilo...")
                            tech_prompt = create_technical_prompt(user_request, w, h, d, os.environ["OPENAI_API_KEY"])

                            # Renderização
                            status_container.write(f"🎨 Pintando pixels (M4 GPU em ação)...")
                            new_img, edges = render_room(tech_prompt, img_path, strength=strength)

                            # Finalização
                            save_path = save_generated_image(new_img)
                            status_container.update(label="Projeto Concluído!", state="complete", expanded=False)

                            st.image(new_img, caption="Proposta WallGen", use_container_width=True)

                            # Feedback do Prompt gerado (Debug oculto)
                            with st.expander("Ver Prompt Técnico Gerado"):
                                st.code(tech_prompt)

                        except Exception as e:
                            status_container.update(label="Erro na Obra", state="error")
                            st.error(f"Detalhe do erro: {e}")

                    with tabs[1]:
                        st.caption("Visão Computacional (O que a IA enxerga das paredes)")
                        if 'edges' in locals():
                            st.image(edges, use_container_width=True)
            else:
                st.info("👈 Faça o upload de uma foto para começar.")