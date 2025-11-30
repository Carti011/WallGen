import os
from openai import OpenAI


def generate_blueprint(user_prompt, width, height, depth, api_key):
    """
    Função pura de Backend.
    Recebe os dados do formulário e retorna o prompt técnico.
    """
    if not api_key:
        raise ValueError("API Key não fornecida.")

    client = OpenAI(api_key=api_key)

    # Prompt do Sistema: Define a persona do Arquiteto
    system_instruction = f"""
    You are an expert Interior Design AI Architect.
    Your goal is to convert a user request into a HIGHLY DETAILED prompt for Stable Diffusion ControlNet.

    CONSTRAINTS:
    - Wall Dimensions: Width {width}m, Height {height}m.
    - Available Depth: {depth}m (Do not suggest furniture deeper than this).
    - Style: Photorealistic, 8k, interior design photography, cinematic lighting.

    OUTPUT FORMAT:
    Return ONLY the prompt string in English. No conversational text.
    Include keywords for lighting, texture, and furniture scale appropriate for the dimensions.
    """

    try:
        # Chamada à API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"User Request: {user_prompt}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        # Retorna o erro como string para ser tratado no frontend
        raise RuntimeError(f"Erro na comunicação com OpenAI: {e}")