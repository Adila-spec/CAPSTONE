#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import html
import streamlit as st 
import pickle

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer


# # classification
# import numpy as np
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.dummy import DummyClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split

# # Regression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import BaggingClassifier

# # topic modelling
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation



# nltk.download('wordnet')
# from nltk.corpus import opinion_lexicon 
# nltk.download('vader_lexicon')
# from nltk.tokenize import word_tokenize
# nltk.download('opinion_lexicon')
# from nltk.corpus import subjectivity
# from nltk.sentiment import SentimentAnalyzer
# from nltk.sentiment.util import *
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from textblob import TextBlob



def clean(text):
    # here is the part converting html escapes like &amp; to characters.
    text = html.unescape(text)
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text) # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_punctuation(text, exclude=None):
    tw_punct = set(punctuation) # speeds up comparison
    tw_punct = tw_punct - exclude
    return ''.join([i for i in text if i not in tw_punct])

def tokenize(text):
    # lowercase as well
    ret = text.lower()
    ret = ret.split(' ')
    # remove potentially empty tokens
    return [i for i in ret if i != '']

def remove_stopwords(tokens, exclude):
    sw = set(stopwords.words("english"))
    sw = sw - exclude
    return [i for i in tokens if i not in sw]
    

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(i) for i in tokens]


# df = pd.read_csv(r"C:\Users\sheri\Downloads\RoeVsWade_tweets_200000.csv")

# df = df.dropna()
# def polarity_score(text):
#     blob = TextBlob(text)
#     return blob.sentiment.polarity

# polarity_scores = list()
# for index, row in df.iterrows():
#     tweet = row['Tweet_text']
#     score = polarity_score(tweet)
#     polarity_scores.append(score)
    
    
# df['Polarity'] = polarity_scores
# df['User_description_cleaned'] = df['User_description'].copy()
# df['User_description_cleaned'] = df['User_description_cleaned'].apply(clean)
# df['lammetaized_tokens'] = df['User_description_cleaned'].apply(remove_punctuation, exclude=set()).apply(tokenize).apply(remove_stopwords, exclude=set(['not', 'never', 'nor'])).apply(lemmatize)
# df['User_description_cleaned'] = df['lammetaized_tokens'].apply(lambda x: ' '.join(x))
# df = df[df['lammetaized_tokens'].str.len() > 5]
# df = df[df['Polarity'] != 0.0]

# polarity_scores = list()
# for index, row in df.iterrows():
#     polarity = row['Polarity']
#     if polarity > 0:
#         polarity_scores.append(1)
#     else:
#         polarity_scores.append(0)
# df['Polarity'] = polarity_scores
# df['Polarity'].value_counts()
# X_train, X_test, Y_train, Y_test = train_test_split(df['User_description_cleaned'],


tfidf = TfidfVectorizer(min_df = 5, ngram_range=(1,1))


with open("RoeVsWadeModel.ml", 'rb') as file:
    model = pickle.load(file)

with open("tfidf.pickle", 'rb') as file:
    tfidf = pickle.load(file)

def quickClassify(tweet):
    tweet = pd.Series(tweet)
    tweet = tweet.apply(remove_punctuation, exclude=set()).apply(tokenize).apply(remove_stopwords, exclude=set(['not', 'never', 'nor'])).apply(lemmatize)
    n_tokens = tweet.str.len().iloc[0]
    
    if n_tokens > 6:
        tweet = tweet.apply(lambda x: ' '.join(x))
        tweet_test_tf = tfidf.transform(tweet)
        sentiment_pred = model.predict(tweet_test_tf)
        sentiment_proba = model.predict_proba(tweet_test_tf)
        if sentiment_pred == 1:
            return ("Positive", sentiment_pred, sentiment_proba)
        else:
            return ("Negative", sentiment_pred, sentiment_proba)
    else:
        return ("The user profile/description is too short", None, None)




st.title("Row Vs Wade Overturn Sentiment Predictor")
fakeTweet = st.text_input("Input Tweet: ", key="tweet")

if st.button('Make Prediction'):
    prediction = quickClassify(fakeTweet)
    if prediction[1]==None:
        st.write(prediction[0])
    else:
        st.write((
            f"This tweet seems {prediction[0]} "
            f"(prediction: {prediction[1]}) "
            f"(confidence: {prediction[2]})"
        ))



# In[ ]:




