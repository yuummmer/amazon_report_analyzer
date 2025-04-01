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
            st.subheader("📄 Extracted Sample")
            st.write(documents[0].page_content[:800] + "...")

            vectorstore = create_vectorstore_from_texts(documents, uploaded_file.name)

            # Full text for splitting
            full_text = " ".join(doc.page_content for doc in documents)
            chunk_size = 3000
            text_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

            # 💡 Filter chunks based on keyword presence
            keywords = [kw.strip().lower() for kw in focus_terms.split(",")] if focus_terms else None

            if keywords:
                relevant_chunks = [
                    chunk for chunk in text_chunks
                    if any(keyword in chunk.lower() for keyword in keywords)
                ]
                st.markdown(f"🔍 Using {len(relevant_chunks)} of {len(text_chunks)} chunks containing focus keywords.")
            else:
                relevant_chunks = text_chunks

            # 🔁 Guided Summarization
            summary = summarize_chunks_with_keywords(relevant_chunks, keywords)

            st.subheader("📝 Summary")
            st.write(summary)

            # 💬 Optional Q&A
            st.subheader("💬 Ask a Question About the Report")
            query = st.text_input("Type your question here:", placeholder="e.g., What are Amazon's logistics goals this year?")

            if query:
                with st.spinner("Searching for an answer..."):
                    try:
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
            st.error("No content could be extracted from the PDF.")
