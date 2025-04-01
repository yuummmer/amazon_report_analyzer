import ollama

def summarize_text(text, model= "mistral"):
    prompt = f"Summarize this text clearly and concisely:\n\n{text}"
    response = ollama.chat(model=model, messages = [{"role": "user", "content": prompt}])
    return response['message']['content']