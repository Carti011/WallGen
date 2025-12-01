from src.config.settings import load_settings
from src.ui.sidebar import render_sidebar
from src.ui.dashboard import render_dashboard

if __name__ == "__main__":
    # Carrega Configurações
    settings = load_settings()

    # Renderiza Sidebar
    render_sidebar()

    # Renderiza Dashboard Principal
    render_dashboard()