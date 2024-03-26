import sys
from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib
import preprocess

app = Flask(__name__)

# Running preprocessing and loading the model before the server starts
with app.app_context():
    preprocess.load_model_tokenizer()

if __name__ == "__main__":
    app.run()

    
@app.route("/status", methods=['GET'])
def model_status():
    status = {"status" : "alive"}
    return jsonify(status)

@app.route('/')
def man():
    return render_template('/home.html')

@app.route("/predict-web", methods=['POST'])
def web_prediction():
    question = [str(request.form['question'])]
    answer = preprocess.predict_question(question)
    return render_template("/result.html", data = answer)

@app.route("/predict", methods=['POST', 'GET'])
def api_prediction():
    question = [str(request.data)]
    # Getting query parameter
    query_question = str(request.args.get('question'))
    # Checking if query param is none
    if query_question is not None:
        question = [query_question]
    answer = preprocess.predict_question(question)
    return jsonify(answer)