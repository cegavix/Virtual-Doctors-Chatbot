from flask import Flask, request, jsonify, render_template

from Transfer_learning_Chatbot.BERT.bert_main import get_prediction

app = Flask(__name__)

# Go to homepage
@app.get('/')
def index_get():
    return render_template('base.html') # Render the HTML file

@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_prediction(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)