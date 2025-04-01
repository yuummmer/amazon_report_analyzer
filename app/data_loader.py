import pandas as pd
import streamlit as st

@st.cache_data
def load_word_data():
    """
    Loads the cleaned report words dataset.
    Update the file path if you're using .feather instead of .csv.
    """
    try:
        df = pd.read_csv("data/top10_words_by_year.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()
