from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def create_dataloaders(train_texts, val_texts, train_labels, val_labels, batch_size=16):
    train_dataset = TweetDataset(train_texts, train_labels)
    val_dataset = TweetDataset(val_texts, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
