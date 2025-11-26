import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# adding dataset
data = pd.read_excel("./data/twiter_sentiment.xlsx")
df = pd.DataFrame(data)

# CLEAN DATAS

# remove unused columns
df = df.drop(columns=["none", "none.1"])
# check null rows
print(df.isna().sum())
# remove rows with NaN values
df = df[df["Text"].notna()]

# data balancing chart
qtdSentiments = []
labels = df['Sentiment'].unique()
for i in labels:
    qtd = df['Sentiment'].value_counts()[i]
    qtd = int(qtd)
    qtdSentiments.append(qtd)
print(qtdSentiments)
