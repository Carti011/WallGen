from openai import OpenAI


def create_technical_prompt(user_request, width, height, depth, api_key):
    """
    Transforma o pedido do usuário em um prompt técnico com PESOS (Weighting) para garantir que os objetos apareçam.
    """
    if not api_key:
        raise ValueError("API Key ausente.")

    client = OpenAI(api_key=api_key)

    system_role = """
    You are a Senior Interior Design AI specialized in Stable Diffusion Prompt Engineering.
    Your goal is to write a prompt that forces the AI to INSERT specific furniture into an existing room.

    CRITICAL TECHNIQUE - PROMPT WEIGHTING:
    You MUST identify the main object the user wants (e.g., "desk", "sofa", "computer") and apply a weight syntax like (object:1.4).
    1.1 = subtle emphasis
    1.3 = strong emphasis
    1.5 = very strong emphasis (use this for the main requested object)

    Structure:
    "(Main Object:1.5), (Secondary Object:1.2), [Style keywords], [Lighting & Atmosphere], [Room Context]"

    Example:
    User: "A gaming pc on a desk"
    Output: "(high end gaming computer setup:1.5), (modern black desk:1.3), neon rgb lighting, cinematic shot, 8k, hyperrealistic, placed in a modern room"
    """

    user_content = f"""
    Room Context: {width}m x {height}m.
    User Request: "{user_request}"

    Output ONLY the English prompt string.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_content}
            ],
            temperature=0.6
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar prompt: {e}"