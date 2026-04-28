import os
from google import genai

# Get API key from environment (Streamlit Secrets)
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

def explain_bias(rates, di):
    prompt = f"""
    These are fairness results:
    Group rates: {rates}
    Disparate Impact: {di}

    Explain if there is bias and what it means in simple terms.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return response.text