import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import bigrams
from collections import Counter
import streamlit as st
import nltk

# Ensure nltk data is downloaded
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        print("Punkt tokenizer found!")
    except LookupError:
        print("Punkt tokenizer not found. Downloading...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
        print("Stopwords corpus found!")
    except LookupError:
        print("Stopwords corpus not found. Downloading...")
        nltk.download('stopwords')

# Call download function
download_nltk_data()

# Prepare stopwords and stemmer
stop_words = stopwords.words('english')
porter = PorterStemmer()

# Clean text
def clean_text(i_text):
    step_1 = i_text.lower()
    step_2 = ''.join([char for char in step_1 if char not in string.punctuation])
    return step_2

# Tokenize, remove stopwords, and stem
def extract_tokens(i_text):
    step_1 = word_tokenize(i_text)
    step_2 = [word for word in step_1 if word not in stop_words]
    step_3 = [porter.stem(word) for word in step_2]
    return step_3

# Plot word cloud
def plot_wordcloud(text):
    fig, ax = plt.subplots()
    wordcloud = WordCloud(max_words=500, height=800, width=1500, background_color="black").generate(text)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis('off')
    ax.set_title('Wordcloud of Tokens')
    st.pyplot(fig)

# Plot bigrams
def plot_bigrams(tokens):
    bigram_counts = Counter(bigrams(tokens))
    bigram_df = pd.DataFrame(bigram_counts.items(), columns=['Bigram', 'Count']).sort_values(by='Count', ascending=False)
    fig, ax = plt.subplots()
    bigram_df.head(20).plot(kind='bar', x='Bigram', y='Count', ax=ax)
    ax.set_title('Top 20 Bigrams')
    ax.set_xticklabels([' '.join(bigram) for bigram in bigram_df.head(20)['Bigram']], rotation=45)
    st.pyplot(fig)

# Preprocess the dataframe
def preprocess_data(df):
    df['text_clean'] = df['text'].apply(clean_text)
    df['tokens'] = df['text_clean'].apply(extract_tokens)
    df['text_tokens'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    # Debugging prints
    st.write("Sample data after preprocessing:")
    st.write(df.head())
    
    # Generate visualizations after preprocessing
    text = " ".join(df['text_tokens'])
    plot_wordcloud(text)
    
    all_tokens = [token for sublist in df['tokens'] for token in sublist]
    plot_bigrams(all_tokens)
    
    return df
