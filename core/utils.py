import torch
import os

def get_device_status():
    """Verifica disponibilidade do Apple Silicon (MPS)."""
    if torch.backends.mps.is_available():
        return "mps", "Apple Silicon (MPS): ATIVO 🚀", "success"
    else:
        return "cpu", "MPS INATIVO (CPU) ⚠️", "error"


def save_uploaded_file(uploaded_file):
    """Salva o arquivo de upload no disco e retorna o caminho."""
    try:
        folder = "temp_data/uploads"
        os.makedirs(folder, exist_ok=True)

        save_path = os.path.join(folder, "input.jpg")

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return save_path
    except Exception as e:
        raise RuntimeError(f"Erro ao salvar arquivo: {e}")