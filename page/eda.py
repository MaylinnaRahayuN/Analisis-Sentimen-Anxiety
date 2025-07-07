import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk import bigrams
from collections import Counter
import pandas as pd
import nltk

# Download necessary nltk data
nltk.download('punkt')

def plot_label_distribution(df):
    st.subheader("Label Distribution")
    fig, ax = plt.subplots()
    df['label'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Label Distribution')
    st.pyplot(fig)


def eda_page(df):
    st.title("Exploratory Data Analysis")
    
    if df is not None:
        # Display basic statistics
        st.subheader("Basic Statistics")
        st.write(df.describe(include='all'))
        
        # Plot label distribution
        plot_label_distribution(df)
        
    else:
        st.write("No dataset uploaded.")
