# 🛍️ RaayeAI
**Roman Urdu Product Review Sentiment Analyzer**

🔗 **Live Demo:** https://raayeai-sentiment1.streamlit.app/
🤗 **Model:** [codeZ1234/RaayeAI-mbert](https://huggingface.co/codeZ1234/RaayeAI-mbert)

---

## About
NLP-based sentiment analysis for code-mixed Roman Urdu product reviews scraped from Daraz.pk. Evolved from classical ML (V1) to fine-tuned Multilingual BERT (V2).

---

## 📊 Dataset
- 16,990 real Daraz.pk reviews
- Code-mixed Roman Urdu + English
- 3 classes: Positive, Neutral, Negative
- Class imbalance handled via resampling

---

## Project Evolution

### V1 — TF-IDF + Logistic Regression
- Classical ML baseline approach
- **Performance:** Negative: 93.6% | Neutral: 92.6% | Positive: 90.7%
- 📓 [View V1 Notebook](https://github.com/codebyzahra/RaayeAI/blob/main/v1_tfidf/RaayeAI_v1_TF_IDF.ipynb)

### V2 — mBERT (Current) ⭐
- Fine-tuned Multilingual BERT
- Handles Roman Urdu slang, emojis, code-mixing
- **Performance:** Negative: 97% | Neutral: 95% | Positive: 95%
- **Overall Accuracy: 96% | ROC-AUC: 0.99**
- 📓 [View V2 Notebook](https://github.com/codebyzahra/RaayeAI/blob/main/v2_mbert/RaayeAI_v2_mBERT.ipynb)

---

## Features
- Live Roman Urdu review analysis
- Confidence score + progress bar
- Seller action suggestions
- Session history tracking
- Custom Roman Urdu slang normalization

---

## Tech Stack
| V1 | V2 |
|----|----|
| Scikit-learn | HuggingFace Transformers |
| TF-IDF | mBERT (bert-base-multilingual-cased) |
| Logistic Regression | PyTorch |
| Streamlit | Streamlit |

---

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```