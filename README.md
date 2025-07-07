# SentimentAnalysis

This project performs sentiment analysis on Twitter data using a hybrid approach:
- **Text preprocessing**: Tokenization, stopword removal, stemming
- **Embedding**: Optional BERT embeddings using Hugging Face Transformers
- **Modeling**: Logistic Regression trained on labeled tweet samples
- **Twitter API**: Tweets collected via Tweepy

### Features
- Real-time tweet collection
- Custom preprocessing pipeline (NLTK-based)
- Model built from scratch (not just using a pre-trained classifier)
- Supports classical ML and deep learning

### Stack
- Python
- Tweepy
- NLTK
- scikit-learn
- Transformers (optional)
- Pandas, NumPy

---

### How to Run

```bash
pip install -r requirements.txt
python SentimentAnalysis.py
