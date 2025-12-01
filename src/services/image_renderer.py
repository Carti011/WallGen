import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import streamlit as st


@st.cache_resource
def load_engine(device="mps"):
    """
    Carrega o motor Img2Img (Imagem para Imagem) + ControlNet.
    """
    print(f"--- 🏗️ CARREGANDO ENGINE IM2IMG NO {device.upper()} ---")

    # Carrega o "Olho" (ControlNet Canny)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        use_safetensors=True
    )

    # Carrega o "Pintor" (Stable Diffusion Img2Img)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    )

    # Otimizações
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.vae.to(dtype=torch.float32)
    pipe.enable_attention_slicing()

    return pipe


def make_canny_condition(image):
    """Cria o esqueleto (linhas) da imagem."""
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def render_room(prompt, input_path, strength=0.75):
    """
    strength: Quão criativa a IA pode ser?
    0.0 = Imagem original idêntica.
    1.0 = Ignora imagem original.
    0.75 = Bom equilíbrio para adicionar móveis mantendo o quarto.
    """
    pipe = load_engine()

    # Prepara as imagens
    original_img = Image.open(input_path).convert("RGB").resize((512, 512))
    control_img = make_canny_condition(original_img)

    # Geração
    result = pipe(
        prompt,
        image=original_img,
        control_image=control_img,
        negative_prompt="cartoon, painting, illustration, distorted, bad perspective, worst quality, low quality",
        strength=strength,
        controlnet_conditioning_scale=1.0, 
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    return result, control_img