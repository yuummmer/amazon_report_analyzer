import streamlit as st
from data_loader import load_word_data
from visuals import plot_top_words
from app.llm_summarizer import summarize_text

st.set_page_config(page_title="Amazon Annual Report Analyzer", layout="centered")

st.title("\U0001F4CA Amazon Annual Report Analyzer")
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

text_input = st.text_area("Paste a section of an Amazon 10-K report:")
if st.button("Summarize"):
    if text_input:
        with st.spinner("Summarizing..."):
            summary = summarize_text(text_input)
            st.success("Done!")
            st.markdown("### ✨ Summary")
            st.write(summary)
    else:
        st.warning("Please paste some text first.")