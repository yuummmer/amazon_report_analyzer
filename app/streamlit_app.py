import streamlit as st
from data_loader import load_word_data
from visuals import plot_top_words

st.set_page_config(page_title="Amazon Annual Report Analyzer", layout="centered")

st.title("\U0001F4CA Amazon Annual Report Analyzer")
st.markdown("Explore Amazon's annual reports using word frequencies and AI.")

# Load dataset
df = load_word_data()

# Year selector
years = sorted(df["year"].unique())
selected_year = st.selectbox("Select a year to explore:", years)

# Show top 10 words
fig = plot_top_words(df, selected_year)
st.plotly_chart(fig, use_container_width=True)
