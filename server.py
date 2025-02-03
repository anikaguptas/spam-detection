from flask import Flask, request, jsonify
from flask_cors import CORS 
import joblib


app = Flask(__name__)


CORS(app)

mnb_model = joblib.load('mnb_model.pkl')  
cv = joblib.load('cv_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Spam Detection API is running!. Msg from Anika'}), 200
 
# Define a route for checking spam
@app.route('/check_spam', methods=['POST'])
def check_spam():
    try:
        data = request.get_json()
        email_text = data.get('email', '')
        if not email_text:
            return jsonify({'error': 'Email text is required.'}), 400
        
        email_transformed = cv.transform([email_text])
        prediction = mnb_model.predict(email_transformed)
        print("prediction is " + str(prediction))
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
