import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import streamlit as st


@st.cache_resource
def load_engine(device="mps"):
    """
    Carrega o motor Img2Img + ControlNet MLSD (Especialista em Arquitetura).
    """
    print(f"--- 🏗️ CARREGANDO ENGINE ARQUITETÔNICA (MLSD) NO {device.upper()} ---")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-mlsd",
        use_safetensors=True
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.vae.to(dtype=torch.float32)
    pipe.enable_attention_slicing()

    return pipe


def get_mlsd_lines(image):
    """
    Simula o algoritmo MLSD usando Visão Computacional Clássica (OpenCV).
    Extrai apenas linhas retas estruturais (paredes, teto, chão),
    ignorando bagunça e texturas orgânicas.
    """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,  # Precisa de 80 votos para ser linha
        minLineLength=100,  # Linha tem que ter min 100px (ignora rabiscos)
        maxLineGap=10  # Permite pequenas falhas na linha
    )

    line_map = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_map, (x1, y1), (x2, y2), (255, 255, 255), 3)

    return Image.fromarray(line_map)


def render_room(prompt, input_path, strength=0.65):
    """
    Renderiza usando MLSD simulado para manter a geometria perfeita.
    """
    pipe = load_engine()

    # Prepara as imagens
    original_img = Image.open(input_path).convert("RGB").resize((512, 512))

    control_img = get_mlsd_lines(original_img)

    negative_prompt = (
        "low quality, blurry, distortion, worst quality, cartoon, "
        "curved walls, slanted walls, distorted perspective, "
        "floating furniture, bad anatomy, messy, cluttered, text, watermark, "
        "window bars, broken geometry, weird artifacts"
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