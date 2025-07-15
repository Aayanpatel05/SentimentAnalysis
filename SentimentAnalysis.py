from data_preprocessing import load_and_prepare_data
from bert_embeddings import create_dataloaders
from train_model import train_evaluate_tfidf_model, train_evaluate_bert_model

# Path to your CSV file
csv_path = 'tweets_sentiments_dataset.csv'

# Step 1: Load and preprocess data
clean_texts, labels = load_and_prepare_data(csv_path)

# Step 2: Train & evaluate TF-IDF + Logistic Regression model
print("\nTraining TF-IDF + Logistic Regression Model...")
train_evaluate_tfidf_model(clean_texts, labels)

# Step 3: Create BERT DataLoaders
print("\nPreparing BERT Dataloaders...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    clean_texts, labels, test_size=0.2, random_state=42)

train_loader, val_loader = create_dataloaders(train_texts, val_texts, train_labels, val_labels)

# Step 4: Fine-tune and evaluate BERT model
print("\nTraining BERT Model...")
train_evaluate_bert_model(train_loader, val_loader)
