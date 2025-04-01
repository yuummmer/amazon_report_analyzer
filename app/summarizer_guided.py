import openai
import os

def summarize_text_with_keywords(text, top_keywords, model="gpt-3.5-turbo"):
    """
    Summarize the text using LLM, emphasizing the provided top keywords.
    """
    keyword_str = ", ".join(top_keywords)
    prompt = f"""
    You are a strategic analyst reading Amazon annual reports to look for patterns.
    Based on the language, recurring themes, and emphasis areas in the document,
    predict what areas Amazon is likely to expand into or prioritize in the next 1–2 years.

    Use specific signals from the text — e.g., increased mentions of services, logistics, sustainability, partnerships, etc.
    Your goal is to infer *strategic direction* — such as product areas, business models, or market expansion.

    Focus Keywords: {keyword_str}

    Report Context:
    {text}

    Forecast:
    """

    try:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"[Error generating forecast]: {str(e)}"


def summarize_chunks_with_keywords(chunks, keywords=None, model="gpt-3.5-turbo"):
    """
    Summarize a list of text chunks using optional keywords to guide the focus.
    """
    summaries = []

    for i, chunk in enumerate(chunks):
        if keywords:
            # Turn list of keywords into a comma-separated string
            keyword_str = ", ".join(keywords)
            prompt = f"Summarize this part of an Amazon annual report, focusing on the following terms: {keyword_str}\n\n{chunk}"
        else:
            prompt = f"Summarize this part of an Amazon annual report clearly and concisely:\n\n{chunk}"

        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            summaries.append(response.choices[0].message.content)
        except Exception as e:
            summaries.append(f"[Error summarizing chunk {i}]: {str(e)}")

    return "\n\n".join(summaries)
