import os
from openai import OpenAI


def analyze_depth_constraints(depth_meters):
    """
    Traduz medidas físicas (metros) em restrições visuais (adjetivos em inglês)
    para que o Stable Diffusion entenda a limitação de espaço.
    """
    try:
        d = float(depth_meters)
    except:
        return "standard size furniture"

    if d <= 0.4:
        return "ultra-slim profile, wall-mounted, shallow depth, compact design, minimal projection from wall"
    elif d <= 0.8:
        return "standard depth, moderate size, functional fit"
    elif d <= 1.5:
        return "deep seating, spacious arrangement, large furniture piece"
    else:
        return "massive scale, room-filling furniture, extensive layout, luxury spacing"


def create_technical_prompt(user_request, width, height, depth, api_key):
    """
    Gera um prompt técnico com 'Constraint Injection' para respeitar a profundidade.
    """
    if not api_key:
        raise ValueError("API Key ausente.")

    client = OpenAI(api_key=api_key)

    # Análise de Restrição Física
    depth_keywords = analyze_depth_constraints(depth)

    # System Role Especializado em Física Visual
    system_role = """
    You are a Senior Interior Design AI specialized in Photorealistic Rendering.
    Your task is to convert a user request into a Technical Prompt for Stable Diffusion ControlNet.

    CRITICAL PROTOCOL - SPATIAL AWARENESS:
    You have received specific 'Depth Keywords' calculated from real measurements. 
    You MUST incorporate these adjectives into the description of the main furniture to ensure it fits the room physically.

    PROMPT STRUCTURE:
    "(Main Object:1.4), (Depth Adjectives:1.3), [Material & Texture], [Lighting], [Room Context], 8k, photorealistic, interior design photography"

    EXAMPLES:
    - If user wants a "desk" and depth is "shallow": "(modern desk:1.4), (ultra-slim profile against wall:1.3), compact depth..."
    - If user wants a "sofa" and depth is "deep": "(luxury sofa:1.4), (deep seating lounge style:1.3), spacious..."
    """

    user_content = f"""
    CONTEXT DATA:
    - Wall Dimensions: {width}m x {height}m
    - Available Depth: {depth}m (Physical Limit)
    - Calculated Depth Adjectives: "{depth_keywords}"

    USER REQUEST: "{user_request}"

    OUTPUT:
    Return ONLY the final English prompt string.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_content}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar prompt: {e}"