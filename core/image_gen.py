import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import streamlit as st

# Configuração de Cache
@st.cache_resource
def load_pipeline():
    """
    Carrega o Stable Diffusion e ControlNet em Float32 (Máxima Compatibilidade MPS).
    """
    print("--- 📥 CARREGANDO MODELOS (MODO PRECISÃO FP32) ---")

    # Carrega ControlNet
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        use_safetensors=True
    )

    # Carrega Stable Diffusion
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    )

    # Otimizações para o Apple Silicon (M4)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to("mps")

    pipe.enable_attention_slicing()

    return pipe


def process_canny_edge(image):
    """Cria o mapa de linhas (esqueleto) da parede."""
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)

    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def generate_image(prompt, input_image_path):
    """
    Pipeline principal de geração.
    """
    pipe = load_pipeline()

    original_image = Image.open(input_image_path).convert("RGB")

    image_resized = original_image.resize((512, 512))

    control_image = process_canny_edge(image_resized)

    # Gera a Imagem
    output = pipe(
        prompt,
        image=control_image,
        negative_prompt="low quality, bad quality, sketches, blurry, distortion, weird perspective, artifacts, dark image, underexposed",
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    return output, control_image