# SentimentAnalysis

This project performs sentiment analysis on Twitter data using a **hybrid machine learning pipeline** that supports both classical ML (TF-IDF + Logistic Regression) and deep learning (BERT). It includes preprocessing, model training, evaluation, and flexible configuration.

---

## Features

- **NLTK-based preprocessing** (stopword removal, stemming)
- Two model pipelines:
  - Classical: TF-IDF + Logistic Regression
  - Deep Learning: BERT (fine-tuned)
- **Balanced class handling** using computed class weights
- Integrated **metrics and performance reporting**
- GPU-aware training with PyTorch
---

## Tech Stack

- Python 3.x
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Transformers (Hugging Face)](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- Pandas, NumPy

---

## Project Structure

- data_preprocessing.py
- bert_embeddings.py
- train_model.py
- sentiment_analysis.py
- tweets_sentiments_dataset.csv

---
## Problems & Solutions
- **Problem**: Initial implementation used `BertModel` with a custom classification head. This resulted in lower accuracy and less stable training.
- **Solution**: Switched to `BertForSequenceClassification`, which is purpose-built for classification tasks. It improved performance by using a built-in classification layer, better default initialization, and tighter integration with Hugging Face training tools.
- **Problem**: The model had low recall for positive and negative tweets due to class imbalance.
- **Solution**: Applied class weighting to both Logistic Regression and BERT's loss function.
- **Problem**: BERT fine-tuning led to overfitting, with validation loss increasing after early epochs.
- **Solution**: Added a linear learning rate scheduler and reduced the number of training epochs.
- **Problem**: Lacked visibility into model performance on unseen data, making it hard to detect overfitting or class-specific issues.
- **Solution**: Added a proper validation loop using `.eval()` and `torch.no_grad()`, with full accuracy and classification reporting after each epoch.
---
## Final Results
- **TFIDF Model**: Yielded an accuracy of 80.23%
- **Bert Model**: Yielded an accuracy of 89.40%





