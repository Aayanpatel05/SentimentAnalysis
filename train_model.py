from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_evaluate_tfidf_model(X, Y):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_vectorized, Y, test_size=0.2, stratify=Y, random_state=2)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    
    print("TF-IDF Logistic Regression Accuracy:", accuracy_score(Y_test, predictions))
    print(classification_report(Y_test, predictions))

def train_evaluate_bert_model(X_bert, Y):
    Xb_train, Xb_test, Yb_train, Yb_test = train_test_split(X_bert, Y, test_size=0.2, stratify=Y, random_state=2)
    model_bert = LogisticRegression(max_iter=1000)
    model_bert.fit(Xb_train, Yb_train)
    
    bert_preds = model_bert.predict(Xb_test)
    print("BERT Embeddings Logistic Regression Accuracy:", accuracy_score(Yb_test, bert_preds))
    print(classification_report(Yb_test, bert_preds))
