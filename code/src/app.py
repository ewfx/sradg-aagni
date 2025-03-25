
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the trained model and scaler
with open('EvaluateRiskModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('EvaluateRiskModelScaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the categorical feature mappings used during training
reason_categories = ["Other reason", "Home ownership", "Not home ownership"]
job_categories = ["Mgr", "Other job", "Office worker", "Retail worker", "Business owner", "Technician"]

app = Flask(__name__)

# Route to display the form
@app.route('/')
def home():
    return render_template('index.html', reason_categories=reason_categories, job_categories=job_categories)

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form input
        loan = float(request.form['loan'])
        mortdue = float(request.form['mortdue'])
        value = float(request.form['value'])
        reason = request.form['reason']
        job = request.form['job']
        yoj = float(request.form['yoj'])
        derog = float(request.form['derog'])
        delinq = float(request.form['delinq'])
        clage = float(request.form['clage'])
        ninq = float(request.form['ninq'])
        clno = float(request.form['clno'])
        debtinc = float(request.form['debtinc'])

        # Prepare the feature vector
        features = np.array([[loan, mortdue, value, yoj, derog, delinq, clage, ninq, clno, debtinc, 0]])

        # One-hot encode the categorical columns (Reason and Job) based on predefined categories
        reason_encoded = [1 if reason == r else 0 for r in reason_categories]
        job_encoded = [1 if job == j else 0 for j in job_categories]
        
        # Add the encoded values to the feature vector
        features = np.concatenate([features, [reason_encoded + job_encoded]], axis=1)

        # Preprocess the data (same as during training)
        features_scaled = scaler.transform(features)

        # Make prediction using the model
        prediction = model.predict(features_scaled)
        
        # Return the result
        result = "The customer is likely to default on the loan." if prediction == 1 else "The customer is likely to repay the loan."
        return render_template('index.html', result=result, reason_categories=reason_categories, job_categories=job_categories)
    
    except Exception as e:
        return render_template('index.html', result=f"An error occurred: {str(e)}", reason_categories=reason_categories, job_categories=job_categories)

if __name__ == '__main__':
    app.run(debug=True)
