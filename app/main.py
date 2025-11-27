from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
import spacy

nltk.download("stopwords")
# execute 'python -m spacy download en_core_web_sm' in cmd to download the package
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

model = joblib.load("../src/model/model.pkl")
vectorizer = joblib.load("../src/model/vectorizer.pkl")

app = Flask(__name__)

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join([token.lemma_ for token in nlp(' '.join(words))])

@app.route("/")
def index():
    return render_template("/index.html", result="None")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")
    text = preprocess(text)

    X = vectorizer.transform([text])
    prediction = model.predict(X.toarray())[0]
    prediction = prediction.lower()

    return render_template("/index.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)