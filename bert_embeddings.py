from transformers import BertTokenizer, BertModel
import torch
import numpy as np

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text_list):
    bert_model.eval()
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # [CLS] token
            embeddings.append(cls_embedding)
    return np.array(embeddings)
