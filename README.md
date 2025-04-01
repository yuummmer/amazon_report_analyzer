# Amazon Annual Report Analyzer (RAG-powered)

Amazon Annual Report Analyzer is a Streamlit app that helps you explore trends in Amazon’s annual reports using top keywords, LLM-powered summaries, and a RAG (Retrieval-Augmented Generation) model for document Q&A.

Upload a PDF annual report and dive into AI-generated insights — all grounded in the original source material.

## 🔍 What It Does

### ✅ Yearly Word Trend Visualization
Load pre-cleaned word frequency data for Amazon annual reports

Select a year and instantly view the top 10 words used that year

Interactive visualization using Plotly

### 🧠 Keyword-Guided LLM Summarization
Upload an annual report PDF

Automatically extracts, chunks, and filters text

Generates focused summaries per top keyword from the selected year

Uses OpenAI's GPT-3.5 model for concise theme extraction

### 💬 RAG-based Q&A Interface
Ask custom questions about the uploaded report

Backed by a retrieval-augmented generation (RAG) pipeline

Uses FAISS vectorstore + keyword-aware prompt injection

Answers are generated strictly from the PDF content, ensuring grounded and trustworthy responses

## 🌐 Live App

Check it out here: [Streamlit Cloud App](https://amazonreportanalyzer-wjhbrs6hybvgwyy4ntfbjt.streamlit.app/)

## 🗃️ Project Structure

| Folder/File              | Description                                                                      |
|--------------------------|----------------------------------------------------------------------------------|
| `app/streamlit_app.py`   | Main app script                                                                  |
| `app/data_loader.py`     | Loads and prepares word frequency data                                           |
| `app/visuals.py`         | Plotly-based visualization utilities                                             |
| `app/pdf_utils.py`       | Extracts text from PDFs, splits documents, and creates FAISS vectorstore         |
| `data/`                  | Word frequency datasets                                                          |
| `requirements.txt`       | App dependencies                                                                 |
| `README.md`              | You're here! 😄                                                                  |


## 💡 Inspiration

Inspired by the [TidyTuesday dataset](https://github.com/rfordatascience/tidytuesday) and a curiosity about how Amazon's language evolves over time, this project started as a simple word frequency plot and grew into an interactive app.

## 🛠️ Running Locally

Clone the repository and set up a virtual environment to run the app locally:

```bash
git clone https://github.com/yuummmer/amazon_report_analyzer.git
cd amazon_report_analyzer
python -m venv venv
source venv/bin/activate      # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
