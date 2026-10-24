import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")
st.title("😊 Sentiment Analysis System")
st.write("By ~ Ashish Ranjan..........................")
st.write("Checks whether text is positive or negative, using a Hugging Face model.")


@st.cache_resource          # load the model only once, not on every click
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


classifier = load_model()

tab1, tab2 = st.tabs(["Single text", "Upload CSV"])

# ---- Tab 1: one piece of text ----
with tab1:
    text = st.text_area("Type a sentence or review:", "I really enjoyed this movie!")
    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Please type something first.")
        else:
            result = classifier(text, truncation=True)[0]
            label, score = result["label"], result["score"]
            if label == "POSITIVE":
                st.success(f"😊 Positive ({score:.1%} confident)")
            else:
                st.error(f"😞 Negative ({score:.1%} confident)")

# ---- Tab 2: many reviews from a CSV ----
with tab2:
    file = st.file_uploader("Upload a CSV with a column named 'text'", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("Your CSV needs a column named 'text'.")
        else:
            df = df.dropna(subset=["text"]).head(500)      # limit to 500 rows
            results = classifier(df["text"].astype(str).tolist(), truncation=True)
            df["sentiment"] = [r["label"] for r in results]
            df["confidence"] = [round(r["score"], 3) for r in results]

            st.bar_chart(df["sentiment"].value_counts())
            st.dataframe(df)
            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False),
                "results.csv",
                "text/csv",
            )
