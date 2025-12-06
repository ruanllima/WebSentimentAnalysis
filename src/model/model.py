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

def run():

    # execute 'python -m spacy download en_core_web_sm' in cmd to download the package
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])

    # adding dataset
    data = pd.read_csv("./data/dataset.csv")
    df = pd.DataFrame(data)
    df = df.sample(frac=0.4)

    # CLEAN DATAS
    df = df.drop(columns=["2401", "Borderlands"])
    df = df.rename(columns={"Positive" : "Sentiment", "im getting on borderlands and i will murder you all ," : "Text"})
    df = df.dropna()
    df = df[df["Text"] != "str"]
    df = df[df["Sentiment"] != "str"]
    # remove unused columns
    #df = df.drop(columns=["none", "none.1"])
    # check null rows
    # print(df.isna().sum())
    # remove rows with NaN values
    #df = df[df["Text"].notna()]

    # DEFINE TRAIN AND TEST DATAS
    X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["Sentiment"], random_state=42)

    # DATA PREPROCESSING

    # Tokenization
    #X_train = [phrase.split() for phrase in X_train]
    # X_test = [phrase.split() for phrase in X_test]

    # Lemmatization
    docsTrain = nlp.pipe(X_train, batch_size=2000, n_process=1)
    docsTest = nlp.pipe(X_test, batch_size=2000, n_process=1)
    X_train = [" ".join(token.lemma_ for token in doc) for doc in docsTrain]
    X_test = [" ".join(token.lemma_ for token in doc) for doc in docsTest]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        token_pattern=r'[a-zA-Z]+',
        max_features=50000
    )
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Data balancing by combining SMOTE and ENN.
    # smote_enn = SMOTEENN() 
    # X_train, y_train = smote_enn.fit_resample(X_train, y_train)

    # ALGORITHM IMPLEMENTATION (SVM) 
    algorithm = SVC()
    algorithm.fit(X_train, y_train)

    y_pred = algorithm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


    # ========================== MODEL =============================
    joblib.dump(algorithm   , "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

if __name__ == "__main__":
    run()