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
    app.run(debug=True, host='0.0.0.0', port=8080)

    
@app.route("/ping", methods=['GET'])
def model_ping():
    status = {"status" : "healthy"}
    return jsonify(status)

@app.route("/invocations", methods=['POST'])
def invoke_prediction():
    request_question = request.get_json()
    question = [str(request_question['question'])]
    answer = preprocess.predict_question(question)
    return jsonify(answer)

@app.route('/')
def man():
    return render_template('/home.html')

@app.route("/predict-web", methods=['POST'])
def web_prediction():
    question = [str(request.form['question'])]
    answer = preprocess.predict_question(question)
    return render_template("/result.html", data = answer)