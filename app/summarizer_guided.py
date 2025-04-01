import openai
import os

def summarize_text_with_keywords(text, top_keywords, model="gpt-3.5-turbo"):
    """
    Summarize the text using LLM, emphasizing the provided top keywords.
    """
    keyword_str = ", ".join(top_keywords)
prompt = f"""
    You are a critical analyst reviewing Amazon's annual report through a sociological lens. 
    Your goal is to extract how Amazon justifies its scale, power, and role in society.

    Specifically, identify:
    - Themes related to corporate responsibility, labor practices, or environmental impact
    - Language used to frame Amazon growth, innovation, or global influence
    - Any notable shifts in tone or emphasis on public perception

    Summarize the content below with attention to how Amazon portrays itself — not just what it reports. Keep the summary short - between 200-500 words if possible.

    Text:
    {text}

    Summary:
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
        return f"[Error generating summary]: {str(e)}"


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
            # Fallback to generic summarization
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
