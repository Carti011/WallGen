from openai import OpenAI


def get_style_guidelines(style_name):
    """
    Banco de dados de conhecimento do Agente Decorador.
    """
    styles = {
        "Moderno": "sleek lines, neutral palette (white, beige, grey), glass and steel materials, minimalism, functional furniture",
        "Industrial": "exposed brick, raw concrete walls, leather furniture, black metal accents, edison bulbs, loft style",
        "Escandinavo": "hygge, light wood (oak, ash), white walls, cozy textiles, plenty of natural light, plants",
        "Luxo": "marble floors, gold accents, velvet upholstery, chandelier, dark wood, high contrast, dramatic lighting"
    }
    return styles.get(style_name, styles["Moderno"])


def create_technical_prompt(user_request, width, height, depth, api_key):
    """
    Gera um prompt técnico usando uma Persona de Agente Decorador.
    """
    if not api_key:
        raise ValueError("Chave da API ausente.")

    client = OpenAI(api_key=api_key)

    system_role = """
    You are a World-Class Interior Designer AI (Agent ID).
    Your goal is to curate a scene that is structurally physically possible and aesthetically cohesive.

    STEP 1: ANALYZE THE REQUEST
    Identify the intended style (Modern, Industrial, etc).

    STEP 2: CHECK PHYSICS
    User Depth Limit: {depth} meters.
    - If depth < 0.6m: USE 'wall-mounted', 'floating console', 'slim'.
    - If depth > 1.0m: USE 'deep sofa', 'central island', 'lounge chair'.

    STEP 3: GENERATE PROMPT
    Format: "(Masterpiece, best quality, 8k, interior photography:1.4), (Style Keywords), (Main Furniture with Physics Adjectives), (Lighting), (Texture details)"
    """

    user_content = f"""
    ROOM DIMENSIONS: {width}m width x {height}m height.
    DEPTH CONSTRAINT: {depth} meters.
    USER REQUEST: "{user_request}"

    INSTRUCTION:
    - Do not simply translate. IMPROVE the request with design terminology.
    - If the user asks for something too big for the depth, replace it with a smaller alternative (e.g., Sofa -> Bench).
    - Output ONLY the English prompt.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_role.format(depth=depth)},
                {"role": "user", "content": user_content}
            ],
            temperature=0.6
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar prompt: {e}"