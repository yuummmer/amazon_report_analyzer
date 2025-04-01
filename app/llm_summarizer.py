import openai
import os

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def summarize_text(text):
    prompt = f"Summarize this text clearly and concisely:\n\n{text}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    return response.choices[0].message.content

def summarize_text_chunks(text_chunks, model="gpt-3.5-turbo"):
    summaries = []

    for i, chunk in enumerate(text_chunks):
        prompt = f"Summarize this part of an Amazon annual report clearly and concisely:\n\n{chunk}"
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            summaries.append(response.choices[0].message.content)
        except Exception as e:
            summaries.append(f"[Error summarizing chunk {i}]: {str(e)}")

    return "\n\n".join(summaries)
