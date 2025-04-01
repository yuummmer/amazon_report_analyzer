import streamlit as st
import openai
import os

from data_loader import load_word_data
from visuals import plot_top_words
from summarizer_guided import summarize_chunks_with_keywords
from pdf_utils import (
    get_pdf_text,
    create_vectorstore_from_texts
)

# Set OpenAI API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Amazon Annual Report Analyzer", layout="centered")

st.title("📊 Amazon Annual Report Analyzer")
st.markdown("Curious about what Amazon talks about in its annual reports? Dive into word trends and AI-generated summaries to spot key themes over time.")

# Load dataset
df = load_word_data()

# Year selector
years = sorted(df["year"].unique())
selected_year = st.selectbox("Select a year to explore:", years)

# Show top 10 words
fig = plot_top_words(df, selected_year)
st.plotly_chart(fig, use_container_width=True)

# Add descriptive summary below the chart
top_words_list = (
    df[df["year"] == selected_year]
    .sort_values(by="n", ascending=False)
    .head(10)["word"]
    .tolist()
)

st.caption(
    f"This chart shows the top 10 most frequent words in Amazon's {selected_year} annual report. "
    f"Top terms include: {', '.join(top_words_list)}."
)

# Get top 10 words from selected year for keyword guidance
top_words = (
    df[df["year"] == selected_year]
    .sort_values(by="n", ascending=False)
    .head(10)["word"]
    .tolist()
)

st.markdown("**Top 10 Words This Year:** " + ", ".join(top_words))

st.header("🧠 LLM Summarization (Experimental)")

uploaded_file = st.file_uploader("Upload an Amazon annual report PDF", type=["pdf"])
focus_terms = st.text_input(
    "Optional: Add focus keywords (comma-separated) to guide the summary", 
    help="Example: sustainability, AWS, logistics"
)

if uploaded_file:
    with st.spinner("Processing PDF..."):
        documents = get_pdf_text(uploaded_file)

        if documents:
            # Show sample
            st.subheader("📄 Extracted Sample")
            st.write(documents[0].page_content[:800] + "...")

            # Create vectorstore
            vectorstore = create_vectorstore_from_texts(documents, uploaded_file.name)

            # 🔁 Summarize in chunks
            full_text = " ".join(doc.page_content for doc in documents)
            chunk_size = 3000
            text_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

            # Guided summarization
            keywords = [kw.strip() for kw in focus_terms.split(",")] if focus_terms else None
            summary = summarize_chunks_with_keywords(text_chunks, keywords=keywords)

            st.subheader("📝 Summary")
            st.write(summary)

            # Store vectorstore in session state for Q&A
            st.session_state["vectorstore"] = vectorstore
            st.session_state["summary_text"] = summary

        else:
            st.error("No content could be extracted from the PDF.")

    # 🧠 Ask Questions about the Report
    st.subheader("💬 Ask a Question About the Report")
    query = st.text_input("Type your question here:", placeholder="e.g., What are Amazon's logistics goals this year?")

    if query:
        if "vectorstore" in st.session_state:
            with st.spinner("Searching for an answer..."):
                try:
                    vectorstore = st.session_state["vectorstore"]
                    retriever = vectorstore.as_retriever(search_type="similarity")
                    docs = retriever.get_relevant_documents(query)
                    context = "\n\n".join(doc.page_content for doc in docs)

                    keyword_str = ", ".join(keywords) if keywords else "no specific focus"

                    prompt = f"""
You are an assistant helping analyze Amazon's annual report. Focus especially on the following terms: {keyword_str}.

Use only the context below from the report to answer the question. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""

                    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    answer = response.choices[0].message.content
                    st.markdown("**Answer:**")
                    st.write(answer)

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please upload and process a PDF first.")

    # 🔮 Forecast Section
    st.subheader("🔮 What's Next? LLM Forecast")
    if "summary_text" in st.session_state:
        forecast_prompt = f"""
Based on the following summary of Amazon's {selected_year} annual report, forecast what strategic moves Amazon is likely to make next.
Consider how Amazon positions itself, recurring themes, and shifts in language. Focus on how Amazon expands into new industries or justifies its scale.

Summary of the {selected_year} report:
{st.session_state['summary_text']}

What are Amazon's likely next moves?
"""

        if st.button("Generate Forecast"):
            with st.spinner("Thinking about Amazon's next moves..."):
                try:
                    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": forecast_prompt}],
                        temperature=0.4,
                    )
                    forecast = response.choices[0].message.content
                    st.markdown("**Predicted Strategic Moves:**")
                    st.write(forecast)
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
