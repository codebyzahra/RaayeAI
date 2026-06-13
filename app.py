import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import re
import emoji as emoji_lib

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("codeZ1234/RaayeAI-mbert")
    tokenizer = AutoTokenizer.from_pretrained("codeZ1234/RaayeAI-mbert")
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device('cpu')
model = model.to(device)
model.eval()

def preprocess(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = emoji_lib.demojize(text, delimiters=(" ", " "))
    roman_urdu_map = {
        # existing
    r'\bbht\b': 'bohot', r'\bbhut\b': 'bohot', r'\bnhi\b': 'nahi',
    r'\bthk\b': 'theek', r'\bthik\b': 'theek', r'\bkr\b': 'kar',
    r'\bphr\b': 'phir', r'\bmje\b': 'mujhe', r'\bor\b': 'aur',
    r'\bachi\b': 'acha', r'\bachha\b': 'acha',

    # neutral fixes
    r'\btheek theek\b': 'average',
    r'\btheek\b': 'average',
    r'\btheak\b': 'average',
    r'\bna acha na bura\b': 'average',
    r'\bkhas nahi\b': 'not special',
    r'\bkuch khas nahi\b': 'nothing special',
    r'\bkuch nahi\b': 'nothing special',
    r'\bdrhi\b': 'delivery',
    r'\bpacking\b': 'packaging',
    r'\bkam se kam\b': 'atleast',
    r'\bthora\b': 'thoda',
    r'\bzyada\b': 'zyada',
    r'\bkaam\b': 'kaam',
    r'\bwaisa\b': 'same',
    r'\bjaisa\b': 'as expected',
    
    r'\bbakwas\b': 'terrible',
    r'\bbakwaas\b': 'terrible',
    r'\bbaqwas\b': 'terrible',
    r'\bbkwas\b': 'terrible',
    r'\bbkws\b': 'terrible',
    r'\bbekar\b': 'useless',
    r'\bwaste\b': 'waste',
    r'\bkharab\b': 'bad quality',
    r'\bghатia\b': 'terrible',
    r'\bghatia\b': 'terrible',
    r'\bfaltu\b': 'useless',
    r'\bnakli\b': 'fake',
    r'\bdhoka\b': 'fraud',
    r'\bloot\b': 'scam',

    r'\bbilkul sahi\b': 'perfectly fine',
    r'\bbilkul acha\b': 'very good',
    r'\bbilkul theek\b': 'perfectly fine',
    r'\bkhush hun\b': 'happy satisfied',
    r'\bpehle bhi\b': 'previously also',
    r'\bdobara lunga\b': 'will buy again',
    r'\brecommend\b': 'recommend',

    r'\bdubara order\b': 'will buy again',
    r'\bkroon gi\b': 'will buy again',
    r'\bkroon\b': 'will buy again',
    r'\bbaqwaass\b': 'terrible',
    r'\bkhas nai\b': 'nothing special',
    r'\bsai tha\b': 'average',
    }
    for pattern, replacement in roman_urdu_map.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_suggestion(sentiment):
    return {
        'Positive': "✅ Great review! Consider requesting a public rating.",
        'Neutral': "⚠️ Room for improvement in product/delivery.",
        'Negative': "🚨 Serious issue flagged. Consider reaching out directly."
    }[sentiment]

st.set_page_config(page_title="RaayeAI", page_icon="🛍️")
st.title("🛍️ RaayeAI v2")
st.subheader("Product Review Sentiment Analyzer — Roman Urdu")
st.markdown("---")

review = st.text_area("Paste or type a review:", height=150,
                       placeholder="e.g. bohot acha product hai...")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        clean = preprocess(review)
        inputs = tokenizer(clean, return_tensors='pt',
                          max_length=256, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(outputs.logits, dim=1).item()

        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = label_map[pred]
        confidence = round(max(probs.tolist()) * 100, 2)
        color_map = {'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'}
        emoji_map = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😠'}

        st.markdown(f"### Sentiment: :{color_map[sentiment]}[{emoji_map[sentiment]} {sentiment}]")
        st.metric("Confidence", f"{confidence}%")
        st.info(get_suggestion(sentiment))