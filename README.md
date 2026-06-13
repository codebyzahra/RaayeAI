# 🛍️ RaayeAI
**Roman Urdu Product Review Sentiment Analyzer**

🔗 **Live Demo:** https://raayeai-sentiment1.streamlit.app/

---

## About
NLP-based sentiment analysis system for code-mixed Roman Urdu product reviews scraped from Daraz.pk. Project evolved from classical ML (V1) to fine-tuned Multilingual BERT (V2).

---

## Project Evolution

### V1 — TF-IDF + Logistic Regression
- Classical ML approach
- Trained on 16,000+ real Daraz reviews
- **Model Performance:**
  - Negative: 93.6%
  - Neutral: 92.6%
  - Positive: 90.7%
- 📓 [View V1 Notebook](https://github.com/codebyzahra/RaayeAI/blob/main/v1_tfidf/RaayeAI_v1_TF_IDF.ipynb)

### V2 — mBERT (Current) ⭐
- Fine-tuned Multilingual BERT
- Handles Roman Urdu, slang, emojis
- Model hosted on HuggingFace
- 📓 [View V2 Notebook](https://github.com/codebyzahra/RaayeAI/blob/main/v2_mbert/RaayeAI_v2_mBERT.ipynb)

---

## Features
- Live review sentiment analysis
- Confidence score display
- Seller action suggestions
- Roman Urdu slang preprocessing

---

## Tech Stack
| V1 | V2 |
|----|----|
| Scikit-learn | HuggingFace Transformers |
| TF-IDF | mBERT |
| Logistic Regression | PyTorch |
| Streamlit | Streamlit |

---

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```