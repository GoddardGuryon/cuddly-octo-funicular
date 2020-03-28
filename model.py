#!/usr/bin/env python
# coding: utf-8

# Simple Naive Bayes Spam Classifier
# 
# Spam email classifier based on Naive Bayes theorem, using NLTK and SciKit-Learn
# 
# Data Source: Kaggle (https://www.kaggle.com/balakishan77/spam-or-ham-email-classification/data)


# import and check data
import pandas as pd
import glob
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import string


df = pd.read_csv("emails.csv")

# clean data
df.drop_duplicates(inplace=True)
df.isnull().sum()

# check cleaning
df.shape

# define function to remove stopwords from data
nltk.download('stopwords')

def preprocessor(data):
    """
    remove punctuations and stop words from text
    explanation: in data provided, check for punctuations, get cleaned list of words, feed that list to stopwords
    and return final list
    """
    return [word for word in (''.join([x for x in data if x not in string.punctuation])).split() if word not in stopwords.words('english')]


# convert text to count tokens
counts = CountVectorizer(analyzer=preprocessor).fit_transform(df['text'])

# split the data (I kept 75% for training)
X_train, X_test, y_train, y_test = train_test_split(counts, df['spam'], test_size = 0.25, random_state = 0)

# initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# check if the model works fine
fake_test = model.predict(X_train)
print(classification_report(y_train, fake_test))
print("Confusion Matrix:\n", confusion_matrix(y_train, fake_test))
print("Accuracy:", accuracy_score(y_train, fake_test))

# feed it real data
real_test = model.predict(X_test)

# check final result
print(classification_report(y_test, real_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, real_test))
print("Accuracy:", accuracy_score(y_test, real_test))

# save the model
import pickle
pickle.dump(model, open("final_model.sav", 'wb'))

# uncomment the lines below to get the model file
#model_new = pickle.load(open("final_model.sav", 'rb'))
#print(model_new.score(X_test, y_test))

