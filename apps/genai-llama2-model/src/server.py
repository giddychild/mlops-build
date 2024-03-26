from flask import Flask, jsonify, request, render_template
from preprocess import LSClient, QAChain, embedding_model, load_model

app = Flask(__name__)

# Running preprocessing and loading the model before the server starts
# with app.app_context():
#     preprocess.preprocessing()

ls = LSClient()
qa_chain = QAChain(embedding=embedding_model(), llm=load_model(), ls=ls)

@app.route("/status", methods=['GET'])
def model_status():
    """
    Returns the status of the model as a JSON object.
    Returns:
        A JSON object containing the status of the model.
    """
    status = {"status" : "alive"}
    return jsonify(status)

#@app.route('/')
#def man():
#    return render_template('/home.html')

#@app.route("/predict-web", methods=['POST'])
#def web_prediction():
#    question = [str(request.form['question'])]
#    answer = preprocess.generate_answer(question)
#    return render_template("/result.html", data = answer)

@app.route("/predict", methods=['POST', 'GET'])
def api_prediction():
    """
    Endpoint for making predictions using the QA chain.
    Expects a JSON payload with the following keys:
    - collection: the name of the collection to use for the prediction
    - query: the query to use for the prediction
    Returns a JSON response with the following keys:
    - result: the result of the prediction
    """
    data = request.json

    collection = data.get("collection")
    if collection is None:
        return jsonify({"error": "The param 'collection' is required. "}), 400
    
    if collection not in qa_chain.available_collections:
        return jsonify({"error": "Invalid collection."}), 400
    query = data.get("query")
    if query is None:
        return jsonify({"error": "The param 'query' is required. "}), 400

    result = qa_chain.execute(collection, query)
    return jsonify({"result": result["result"]})

if __name__ == "__main__":
    app.run()