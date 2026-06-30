import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import re
import emoji as emoji_lib

#@st.cache_resource
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

# 1. Modern Page Configuration (Sets wide and clean layout)
st.set_page_config(
    page_title="RaayeAI", 
    page_icon="🛍️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS (Styles buttons and history cards for a sleek look)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .header-banner h1 {
        font-size: 3em;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-banner p {
        font-size: 1.2em;
        margin: 10px 0 0 0;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Stats Bar */
    .stats-bar {
        display: flex;
        justify-content: space-around;
        margin-bottom: 30px;
        gap: 15px;
    }
    
    .stat-card {
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-card-value {
        font-size: 2em;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .stat-card-label {
        font-size: 0.9em;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        width: 100%;
        font-weight: 700;
        padding: 15px 20px;
        border: none;
        font-size: 1.1em;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* History Card */
    .history-card {
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        margin-bottom: 15px;
        border-left: 5px solid #667eea;
        border: 1px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .history-card:hover {
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5em;
        font-weight: 700;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #667eea;
    }
    
    /* Input Area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    .stTextArea textarea:focus {
        border-color: #764ba2;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        margin: 30px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.85em;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Session State Initialization (Tracks recent reviews history)
if "history" not in st.session_state:
    st.session_state.history = []

# Main App Headers - Beautiful Banner
st.markdown("""
    <div class="header-banner">
        <h1>🛍️ RaayeAI </h1>
        <p>✨ Product Review Sentiment Analyzer — Roman Urdu ✨</p>
    </div>
""", unsafe_allow_html=True)

# Stats Bar showing analysis count
col_stat1, col_stat2, col_stat3 = st.columns(3)
with col_stat1:
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">📊</div>
            <div class="stat-card-label">Reviews Analyzed</div>
            <div class="stat-card-value" style="font-size: 1.8em; color: #fff;">{len(st.session_state.history)}</div>
        </div>
    """, unsafe_allow_html=True)

with col_stat2:
    positive_count = sum(1 for item in st.session_state.history if item["sentiment"] == "Positive")
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">😊</div>
            <div class="stat-card-label">Positive Reviews</div>
            <div class="stat-card-value" style="font-size: 1.8em; color: #fff;">{positive_count}</div>
        </div>
    """, unsafe_allow_html=True)

with col_stat3:
    negative_count = sum(1 for item in st.session_state.history if item["sentiment"] == "Negative")
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">😠</div>
            <div class="stat-card-label">Negative Reviews</div>
            <div class="stat-card-value" style="font-size: 1.8em; color: #fff;">{negative_count}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# 4. Tabs Setup
tab1, tab2 = st.tabs(["🔍 Analyze Review", "📜 History Logs"])

with tab1:
    # Divide layout into 2 columns: Left for input, Right for results
    col1, col2 = st.columns([1.5, 1], gap="large")
    
    with col1:
        st.markdown('<p class="section-header">📝 Enter Your Review</p>', unsafe_allow_html=True)
        review = st.text_area(
            "Paste or type a review:", 
            height=200,
            placeholder="e.g. bohot acha product hai... bilkul sahi! 💯",
            label_visibility="collapsed"
        )
        
        # Analyze Button with spacing
        st.write("")
        btn_clicked = st.button("🚀 Analyze Sentiment ✨", use_container_width=True)
        st.write("")

    with col2:
        st.markdown('<p class="section-header">📊 Analysis Result</p>', unsafe_allow_html=True)
        
        if btn_clicked:
            if review.strip() == "":
                st.warning("⚠️ Please enter a review to analyze!")
            else:
                # --- Model Execution Engine ---
                clean = preprocess(review)
                inputs = tokenizer(clean, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1)[0]
                    pred = torch.argmax(outputs.logits, dim=1).item()

                label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                sentiment = label_map[pred] # type: ignore
                confidence = round(max(probs.tolist()) * 100, 2)
                
                emoji_map = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😠'}
                # --------------------------------------

                # Append results to session history
                st.session_state.history.append({
                    "text": review, 
                    "sentiment": sentiment, 
                    "emoji": emoji_map[sentiment],
                    "confidence": confidence
                })

                # Beautiful UI output status indicators
                if sentiment == 'Positive':
                    st.success(f"### {emoji_map[sentiment]} Positive Sentiment")
                elif sentiment == 'Neutral':
                    st.warning(f"### {emoji_map[sentiment]} Neutral Sentiment")
                else:
                    st.error(f"### {emoji_map[sentiment]} Negative Sentiment")
                
                # Display confidence with progress bar
                col_conf1, col_conf2 = st.columns([1, 2])
                with col_conf1:
                    st.metric("Confidence", f"{confidence}%")
                with col_conf2:
                    st.progress(confidence / 100)
                
                # Display suggestion
                st.markdown("---")
                st.info(get_suggestion(sentiment))
        else:
            st.markdown("""
                <div style="text-align: center; padding: 40px 20px; color: #999;">
                    <p style="font-size: 1.2em;">⏳ Awaiting input...</p>
                    <p>Type a review on the left and click <strong>Analyze Sentiment</strong> to get started! 🚀</p>
                </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<p class="section-header">🕒 Recent Reviews History</p>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("📭 No reviews analyzed in this session yet. Start analyzing to see your history here!")
    else:
        # Show summary stats first
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        total_reviews = len(st.session_state.history)
        positive = sum(1 for item in st.session_state.history if item["sentiment"] == "Positive")
        negative = sum(1 for item in st.session_state.history if item["sentiment"] == "Negative")
        neutral = total_reviews - positive - negative
        
        with summary_col1:
            st.metric("Total Reviews", total_reviews)
        with summary_col2:
            st.metric("Positive %", f"{round(positive/total_reviews*100)}%")
        with summary_col3:
            st.metric("Negative %", f"{round(negative/total_reviews*100)}%")
        
        st.markdown("---")
        
        # Loop in reverse order to display the latest review on top
        for idx, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"""
                <div class="history-card">
                    <strong style="font-size: 1.1em;">Review #{total_reviews - idx + 1}</strong><br>
                    <strong>Text:</strong> "{item['text']}" <br>
                    <strong>Sentiment:</strong> {item['emoji']} <span style="color: #667eea; font-weight: 700;">{item['sentiment']}</span><br>
                    <strong>Confidence:</strong> {item['confidence']}%
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>🎯 RaayeAI v2 - Powered by mBERT | Analyzing Roman Urdu Product Reviews</p>
        <p style="font-size: 0.8em;">© 2024 RaayeAI. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)