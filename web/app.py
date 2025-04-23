from flask import Flask, render_template, request, jsonify
import requests
import json
import os

app = Flask(__name__)
RASA_API_URL = "http://rasa:5005/webhooks/rest/webhook"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    # Send message to Rasa
    response = requests.post(
        RASA_API_URL,
        json={"sender": "user", "message": user_message}
    )
    
    # Parse Rasa response
    responses = response.json()
    bot_responses = [resp.get('text', '') for resp in responses]
    
    return jsonify({"responses": bot_responses})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)