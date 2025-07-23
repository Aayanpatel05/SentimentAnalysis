import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from data_preprocessing import load_and_process_data
from bert_embeddings import create_dataloaders
from train_model import train_evaluate_tfidf_model, train_evaluate_bert_model

def main():
    df = load_and_process_data('tweets_sentiments_dataset.csv')

    clean_texts = df['stemmed_content'].tolist()
    raw_texts = df['clean_text'].tolist()
    labels = df['sentiment'].tolist()

    class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

    print("\nTraining TF-IDF + Logistic Regression Model...")
    train_evaluate_tfidf_model(clean_texts, labels)

    print("\nPreparing BERT Dataloaders...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        clean_texts, labels, test_size=0.2, random_state=42, stratify=labels)

    train_loader, val_loader = create_dataloaders(train_texts, val_texts, train_labels, val_labels)

    print("\nTraining BERT Model...")
    train_evaluate_bert_model(train_loader, val_loader, class_weights=class_weights)

if __name__ == "__main__":
    main()
