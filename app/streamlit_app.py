import streamlit as st

from data_loader import load_word_data
from visuals import plot_top_words
from llm_summarizer import summarize_text
from pdf_utils import (
    get_pdf_text,
    create_vectorstore_from_texts
)
# Set OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

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

if uploaded_file:
    with st.spinner("Processing PDF..."):
        documents = get_pdf_text(uploaded_file)

        if documents:
            # Show a sample of the extracted text
            st.subheader("📄 Extracted Sample")
            st.write(documents[0].page_content[:800] + "...")

            # Create vector store from extracted documents
            vectorstore = create_vectorstore_from_texts(documents, uploaded_file.name)

            # Summarize entire document with LLM
            full_text = " ".join(doc.page_content for doc in documents)
            summary = summarize_text(full_text)

            st.subheader("📝 Summary")
            st.write(summary)

            # Future: Q&A interface here
            # query = st.text_input("Ask a question about this report:")
            # if query:
            #     answer = query_document(vectorstore, query)
            #     st.subheader("💬 Answer")
            #     st.write(answer)

        else:
            st.error("No content could be extracted from the PDF.")
