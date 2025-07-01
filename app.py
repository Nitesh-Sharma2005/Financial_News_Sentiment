import streamlit as st
from helper import predict_sentiment

# Page setup
st.set_page_config(page_title="Financial News Sentiment Analyzer", layout="centered")

# Title
st.title("📈 Financial News Sentiment Analyzer")
st.markdown("Analyze the **sentiment** of financial news headlines or short articles. Useful for market mood analysis.")

# Input field
user_input = st.text_area("📝 Enter Financial News Text Below", height=150)

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        sentiment_map = {
            "pos": "🟢 Positive",
            "neg": "🔴 Negative",
            "neu": "🟡 Neutral"
        }

        st.subheader("🔎 Prediction Result")
        st.markdown(f"**Sentiment:** {sentiment_map.get(sentiment, sentiment)}")
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
