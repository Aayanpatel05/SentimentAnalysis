from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import torch
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score


def train_evaluate_tfidf_model(X, Y):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_vectorized, Y, test_size=0.2, stratify=Y, random_state=2)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    
    print("TF-IDF Logistic Regression Accuracy:", accuracy_score(Y_test, predictions))
    print(classification_report(Y_test, predictions))


def train_evaluate_bert_model(train_loader, val_loader, num_labels=3, epochs=3, learning_rate=2e-5):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print("BERT Validation Accuracy:", accuracy)
    return accuracy
