import openai
import os

def summarize_text(text):
    prompt = f"Summarize this text clearly and concisely:\n\n{text}"
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or gpt-4 if you have access
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    return response.choices[0].message.content
