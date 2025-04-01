import openai
import os

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def summarize_text_with_keywords(chunk, keywords, model="gpt-3.5-turbo"):
    keyword_str = ", ".join(keywords) if keywords else "none"
    prompt = f"""
Summarize this section of an Amazon annual report with a focus on the following terms: {keyword_str}

Text:
{chunk}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"[Error summarizing chunk]: {str(e)}"
