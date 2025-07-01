import joblib
import numpy as np
import warnings
import string
import nltk

# Ignore warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data safely
try:
   nltk.download('stopwords')
except LookupError:
   nltk.data.find('stopwords')

try:
    nltk.download('wordnet')
except LookupError:
    nltk.data.find('wordnet')

try:
    nltk.download('omw-1.4')
except LookupError:
    nltk.data.find('corpora/omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load stop words and lemmatizer
stop_words = stopwords.words('english')
wnl = WordNetLemmatizer()

# Load model and vectorizer
model = joblib.load('Sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')


def remove_punct(s):
    return ''.join([i for i in s if i not in string.punctuation])

def remove_stopwords(text):
    return ' '.join([i for i in text.split() if i not in stop_words])

def remove_words_less_than_two_chars(text):
    return ' '.join([i for i in text.split() if len(i) > 2])

def lemmatizing_text(text):
    return ' '.join([wnl.lemmatize(word) for word in text.split()])

def preprocess_text(text):
    text = text.lower()
    text = remove_punct(text)
    text = remove_stopwords(text)
    text = remove_words_less_than_two_chars(text)
    text = lemmatizing_text(text)
    return text

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    arr = vectorizer.transform([preprocessed_text])
    pred = model.predict(arr)
    prob = round(np.max(model.predict_proba(arr)), 3)
    return pred[0], prob
