# Amazon Annual Report Analyzer

Analyze and explore Amazon’s annual reports using word frequencies and a RAG-based LLM system to gain insights.

## 🔍 What It Does

This app processes and visualizes word frequency data from Amazon's annual 10-K reports. It allows users to:

- View the most frequently used words over the years  
- Spot trends and shifts in Amazon’s communication  
- Explore interactive charts and insights  

✨ Stay tuned: LLM-based summarization is coming soon!

## 🌐 Live App

Check it out here: [Streamlit Cloud App](https://amazonreportanalyzer-wjhbrs6hybvgwyy4ntfbjt.streamlit.app/)

## 🗃️ Project Structure

| Folder/File             | Description                                |
|-------------------------|--------------------------------------------|
| `app/streamlit_app.py`  | Main app script                            |
| `app/data_loader.py`    | Loads and prepares word frequency data     |
| `app/visuals.py`        | Visualization utilities                    |
| `data/`                 | Word frequency datasets                    |
| `requirements.txt`      | App dependencies                           |
| `README.md`             | You're here! 😄                             |

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
