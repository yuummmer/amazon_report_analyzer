import streamlit as st
import openai

from data_loader import load_word_data
from visuals import plot_top_words
from llm_summarizer import summarize_text_chunks
from summarizer_guided import summarize_chunks_with_keywords
from retriever import query_document
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
        else:
            st.error("No content could be extracted from the PDF.")
