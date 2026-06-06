import streamlit as st
import joblib
import re
import emoji

# Load models
lr = joblib.load('raayeai_lr_model.pkl')
tfidf = joblib.load('raayeai_tfidf.pkl')

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'http\S+|www\S+', '', text)
    roman_urdu_map = {
        r'\bbht\b': 'bohot', r'\bbhut\b': 'bohot', r'\bnhi\b': 'nahi',
        r'\bthk\b': 'theek', r'\bthik\b': 'theek', r'\bkr\b': 'kar',
        r'\bphr\b': 'phir', r'\bmje\b': 'mujhe', r'\bor\b': 'aur',
    }
    for pattern, replacement in roman_urdu_map.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Suggestions
def get_suggestion(sentiment):
    suggestions = {
        'Positive': "✅ Great review! Consider requesting a public rating from this customer.",
        'Neutral':  "⚠️ Customer seems unsatisfied. Room for improvement in product/delivery.",
        'Negative': "🚨 Serious issue flagged. Consider reaching out to this customer directly."
    }
    return suggestions[sentiment]

# UI
st.set_page_config(page_title="RaayeAI", page_icon="🛍️", layout="centered")
st.title("🛍️ RaayeAI")
st.subheader("Product Review Sentiment Analyzer for Roman Urdu")
st.markdown("---")

review = st.text_area("Paste or type a product review:", height=150, 
                       placeholder="e.g. bohot acha product hai, time pe deliver hua...")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        clean = preprocess(review)
        vector = tfidf.transform([clean])
        pred = lr.predict(vector)[0]
        proba = lr.predict_proba(vector)[0]

        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = label_map[pred]
        confidence = round(max(proba) * 100, 2)

        emoji_map = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😠'}
        color_map = {'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'}

        st.markdown(f"### Sentiment: :{color_map[sentiment]}[{emoji_map[sentiment]} {sentiment}]")
        st.metric("Confidence", f"{confidence}%")
        st.info(get_suggestion(sentiment))