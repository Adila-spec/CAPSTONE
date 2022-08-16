from flask import Flask
from flask import request
from flask import render_template
import joblib
# from flask import url_for
# import pandas as pd

from evaluate import *

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form.get('tweet')
    tweet = preprocess_synopsis(tweet)
    tfidf= joblib.load("tfidf.sav")
    tweet_tf = tfidf.transform(tweet)
    model = joblib.load("RandomForest_model.sav")
    sentiment_pred = model.predict(tweet_tf)
    sentiment_proba = model.predict_proba(tweet_tf)
    if sentiment_pred == 1:
        return "positive: " + str(sentiment_proba)
    else:
        return "negative: " + str(sentiment_proba)
if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')
