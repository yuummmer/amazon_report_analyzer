import plotly.express as px

def plot_top_words(df, year, top_n=10):
    top_words = (
        df[df["year"] == year]
        .sort_values("n", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        top_words,
        x="n",
        y="word",
        orientation="h",
        title=f"Top {top_n} Words in Amazon's {year} Report",
        labels={"n": "Count", "word": "Word"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig

