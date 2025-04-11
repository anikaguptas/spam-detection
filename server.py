from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import string
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import urllib.parse
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

ipqs_key = os.getenv('IPQS_API_KEY')

# Ensure NLTK resources are downloaded only ONCE during setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# Load the trained model and vectorizer once (avoids reloading per request)
mnb_model = joblib.load('mnb_model.pkl')
cv = joblib.load('cv_model.pkl')

# Initialize stopwords and stemmer once (avoids re-initialization per request)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """Applies text preprocessing steps."""
    text = text.lower()  # Convert to lowercase
    text = BeautifulSoup(text, 'html.parser').get_text()  # Remove HTML tags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Spam Detection API is running!. Msg from Anika'}), 200

@app.route('/check_spam', methods=['POST'])
def check_spam():
    try:
        data = request.get_json()
        email_text = data.get('email', '')

        if not email_text:
            return jsonify({'error': 'Email text is required.'}), 400
        
        # Clean and preprocess the email text
        cleaned_email = clean_text(email_text)
        print(cleaned_email)
        # Transform and predict
        email_transformed = cv.transform([cleaned_email])
        prediction = mnb_model.predict(email_transformed)
        print("Prediction:", prediction)

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

class IPQS:
    key = ipqs_key

    def malicious_url_scanner_api(self, url: str, params: dict = {}) -> dict:
        scan_url = f'https://www.ipqualityscore.com/api/json/url/{self.key}/{urllib.parse.quote_plus(url)}'
        print(f"🛰️ Calling IPQS API: {scan_url}")
        response = requests.get(scan_url, params=params)
        print(f"🔧 IPQS Status: {response.status_code}")
        return response.json()

@app.route('/check_url', methods=['POST'])
def check_url():
    try:
        data = request.get_json()
        url = data.get('url', '')
        if not url:
            return jsonify({'error': 'URL is required.'}), 400

        print(f"🔍 Scanning direct URL: {url}")
        ipqs = IPQS()
        result = ipqs.malicious_url_scanner_api(url, {'strictness': 0})
        print(f"📦 IPQS Direct Response:\n{json.dumps(result, indent=2)}")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error in /check_url: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=101010)
    app.run()
