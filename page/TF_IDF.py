# TF_IDF.py
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(df):
    maxfeat = 250
    tfidf = TfidfVectorizer(max_features=maxfeat)
    tfidf_matrix = tfidf.fit_transform(df['text_tokens'])
    return tfidf_matrix, tfidf
