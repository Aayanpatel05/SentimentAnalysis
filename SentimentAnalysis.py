from data_preprocessing import load_and_process_data
from bert_embeddings import get_bert_embeddings
from train_model import train_evaluate_tfidf_model, train_evaluate_bert_model

if __name__ == "__main__":
    file_path = "tweets_sentiments_dataset.csv"
    twitter_data = load_and_process_data(file_path)
    
    # You can try stemming or lemmatized content here:
    X = twitter_data['lemmatized_content'].values
    Y = twitter_data['sentiment'].values
    
    # TF-IDF + Logistic Regression
    train_evaluate_tfidf_model(X, Y)
    
    # BERT embeddings + Logistic Regression
    X_bert = get_bert_embeddings(twitter_data['clean_text'].tolist())
    train_evaluate_bert_model(X_bert, Y)
