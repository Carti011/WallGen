import os
from datetime import datetime


def save_uploaded_image(uploaded_file):
    """Salva a imagem enviada pelo usuário em uma pasta temporária."""
    try:
        folder = "temp_data/uploads"
        os.makedirs(folder, exist_ok=True)

        file_path = os.path.join(folder, "current_input.jpg")

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path
    except Exception as e:
        raise RuntimeError(f"Falha ao salvar arquivo de upload: {e}")


def save_generated_image(image):
    """Salva a imagem gerada pela IA na pasta de saídas."""
    try:
        folder = "temp_data/generated"
        os.makedirs(folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wallgen_{timestamp}.png"
        file_path = os.path.join(folder, filename)

        image.save(file_path)

        return file_path
    except Exception as e:
        raise RuntimeError(f"Falha ao salvar imagem gerada: {e}")