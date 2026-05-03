# app.py
from flask import Flask, render_template, request, jsonify
from model.predict import predict_sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review', '')
    sentiment, confidence = predict_sentiment(review)
    return jsonify({
        'sentiment': sentiment,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)





#create virtual environment if u dont have
#python3.10 -m venv env  


# pip install -r requirements.txt

# run the training script
# python -m model.train 
# python -m model.train

# run the flask app
# python app.py