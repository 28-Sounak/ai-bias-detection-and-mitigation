from google import genai

client = genai.Client(api_key="AIzaSyC4EFa4Clvi1XE1CFEs7jx2key32c177fE")

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