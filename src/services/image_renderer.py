import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import streamlit as st


@st.cache_resource
def load_engine(device="mps"):
    """
    Carrega o motor Img2Img + ControlNet com configurações otimizadas.
    """
    print(f"--- 🏗️ CARREGANDO ENGINE IM2IMG NO {device.upper()} ---")

    # Carrega ControlNet
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        use_safetensors=True
    )

    # Carrega Stable Diffusion Img2Img
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


def render_room(prompt, input_path, strength=0.65):
    """
    Renderiza a imagem misturando a original com o novo prompt.
    strength:
        0.65 = Equilibrio ideal. Permite adicionar móveis sem destruir totalmente o quarto original.
        Se ficar muito fraco (móveis transparentes), aumente para 0.75.
    """
    pipe = load_engine()

    # Prepara as imagens
    original_img = Image.open(input_path).convert("RGB").resize((512, 512))
    control_img = make_canny_condition(original_img)

    negative_prompt = (
        "low quality, bad quality, sketches, blurry, distortion, worst quality, "
        "illustration, cartoon, painting, "
        "floating furniture, bad anatomy, ghost, disappearing objects, "
        "messy, cluttered, text, watermark"
    )

    # Geração
    result = pipe(
        prompt,
        image=original_img,
        control_image=control_img,
        negative_prompt=negative_prompt,
        strength=strength,
        controlnet_conditioning_scale=1.0,
        num_inference_steps=30,
        guidance_scale=8.5
    ).images[0]

    return result, control_img