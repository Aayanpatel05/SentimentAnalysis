import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

port_stem = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

def lemmatize_text(content):
    content = re.sub('[^a-zA-Z]', " ", content.lower())
    tokens = content.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_process_data(file_path):
    twitter_data = pd.read_csv(file_path)
    twitter_data.replace({'sentiment': {'positive': 0, 'neutral': 1, 'negative': 2}}, inplace=True)
    
    # Choose either stemming or lemmatizing, or both for comparison
    twitter_data['stemmed_content'] = twitter_data['clean_text'].apply(stemming)
    twitter_data['lemmatized_content'] = twitter_data['clean_text'].apply(lemmatize_text)
    
    return twitter_data
