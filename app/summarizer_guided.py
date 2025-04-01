import openai

def summarize_text_with_keywords(text, top_keywords, model="gpt-3.5-turbo"):
    """
    Summarize the text using LLM, emphasizing the provided top keywords.
    """
    keyword_str = ", ".join(top_keywords)
    prompt = f"""
    You are an expert analyst reviewing an Amazon annual report. Your goal is to write a clear, concise summary
    of the document while paying special attention to the following important terms:

    {keyword_str}

    Here's the report text:
    {text}
    """

    try:
        import openai
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response["choices"][0]["message"]["content"]

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
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            summaries.append(response["choices"][0]["message"]["content"])
        except Exception as e:
            summaries.append(f"[Error summarizing chunk {i}]: {str(e)}")

    return "\n\n".join(summaries)
