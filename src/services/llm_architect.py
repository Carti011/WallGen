from openai import OpenAI


def create_technical_prompt(user_request, width, height, depth, api_key):
    """
    Transforma o pedido do usuário em um prompt técnico otimizado para ControlNet.
    """
    if not api_key:
        raise ValueError("API Key ausente.")

    client = OpenAI(api_key=api_key)

    system_role = """
    You are a Senior Interior Design AI. 
    Your task is to rewrite user descriptions into a prompt for Stable Diffusion ControlNet (Img2Img).

    CRITICAL INSTRUCTION:
    The goal is 'Virtual Staging'. You must keep the existing structural elements (walls, floors, windows) implies by the user input unless told otherwise.
    Focus the prompt on INSERTING the new furniture requested.

    Structure:
    "[Style description], [New Furniture detailed], [Lighting & Atmosphere], [Maintain existing room structure]"
    """

    user_content = f"""
    Room Dimensions: {width}m x {height}m (Depth available: {depth}m).
    User Request: "{user_request}"

    Output ONLY the English prompt.
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