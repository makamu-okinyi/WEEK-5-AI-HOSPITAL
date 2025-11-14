import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# ----------------------------------------------------------------------
# 1. Initialize the Flask App
# ----------------------------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------------------------
# 2. Load the Model Artifacts (Do this ONCE at startup)
# ----------------------------------------------------------------------
print("--- Loading model artifacts ---")
try:
    model = joblib.load('readmission_model.pkl')
    scaler = joblib.load('data_scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print("✓ Model, scaler, and columns loaded successfully.")
except FileNotFoundError:
    print("Error: Model artifacts not found. Run 'hospital_readmission_ml.py' first.")
    exit()

# ----------------------------------------------------------------------
# 3. Create the Preprocessing Function for NEW data
# ----------------------------------------------------------------------
def preprocess_new_data(json_data):
    """
    Takes raw JSON data from the request and processes it
    to match the 33-feature format the model was trained on.
    """
    
    # Convert JSON to a single-row DataFrame
    df = pd.DataFrame(json_data, index=[0])
    
    # --- Re-create engineered features (must match the original script) ---
    
    # 1. Age groups
    df['age_group'] = pd.cut(df['age'], 
                             bins=[0, 50, 65, 80, 100],
                             labels=['<50', '50-65', '65-80', '80+'])
    
    # 2. Polypharmacy
    df['polypharmacy'] = (df['num_medications'] >= 5).astype(int)
    
    # 3. High comorbidity
    df['high_comorbidity'] = (df['num_comorbidities'] >= 3).astype(int)
    
    # 4. Interaction
    df['age_x_comorbid'] = df['age'] * df['num_comorbidities']

    # 5. Days since last admit
    df['days_since_last_admit'] = np.where(
        df['prior_admissions'] > 0,
        90,  # Use a median/average for live prediction
        365
    )
    
    # 6. Log transforms
    df['log_length_of_stay'] = np.log1p(df['length_of_stay'])
    df['log_distance'] = np.log1p(df['distance_to_hospital'])
    
    # 7. Missing indicators (assume 'False' for a new request)
    df['hemoglobin_missing'] = 0
    df['creatinine_missing'] = 0

    # 8. Gender encoding
    df['gender_encoded'] = (df['gender'] == 'M').astype(int)

    # 9. One-hot encoding
    df = pd.get_dummies(df, columns=['insurance', 'diagnosis', 'discharge_location', 'age_group'],
                         drop_first=True, dtype=int)
    
    # --- Final alignment ---
    # It ensures the new data has the *exact same 33 columns*
    # in the *exact same order* as the training data.
    
    # Add any missing columns from the original training set (and fill with 0)
    df_processed = df.reindex(columns=model_columns, fill_value=0)
    
    # Drop any columns that were not in the original training set
    df_processed = df_processed[model_columns]
    
    return df_processed

# ----------------------------------------------------------------------
# 4. Define API Routes
# ----------------------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    """
    Root endpoint - provides API information
    """
    return jsonify({
        'message': 'Hospital Readmission Prediction API',
        'version': '1.0',
        'endpoints': {
            '/': 'API information (this page)',
            '/health': 'Health check endpoint',
            '/predict': 'POST endpoint for readmission risk prediction'
        },
        'usage': {
            'endpoint': '/predict',
            'method': 'POST',
            'content_type': 'application/json',
            'example_request': {
                'patient_id': '12345',
                'age': 72,
                'gender': 'M',
                'num_comorbidities': 3,
                'num_medications': 6,
                'length_of_stay': 5,
                'hemoglobin': 12.5,
                'creatinine': 1.2,
                'insurance': 'Medicare',
                'distance_to_hospital': 15,
                'prior_admissions': 2,
                'ed_visits_6mo': 1,
                'diagnosis': 'Heart Failure',
                'discharge_location': 'Home'
            }
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    try:
        # Check if model, scaler, and columns are loaded
        if model is None or scaler is None or model_columns is None:
            return jsonify({
                'status': 'unhealthy',
                'error': 'Model artifacts not loaded'
            }), 500
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'features': len(model_columns)
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict_readmission():
    """
    The main API endpoint. It receives patient data,
    processes it, makes a prediction, and returns the risk.
    """
    
    # 1. Get the JSON data from the request
    if not request.json:
        return jsonify({'error': 'No JSON data provided. Please send a POST request with JSON data.'}), 400
    
    json_data = request.json
    
    # 2. Preprocess the data
    try:
        processed_data = preprocess_new_data(json_data)
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Preprocessing failed: {str(e)}'}), 400
    
    # 3. Scale the data
    scaled_data = scaler.transform(processed_data)
    
    # 4. Make prediction
    # model.predict_proba() gives [prob_of_0, prob_of_1]
    # We want the probability of class 1 (readmission)
    risk_probability = model.predict_proba(scaled_data)[0, 1]
    
    # 5. Stratify the risk (using logic from your script)
    if risk_probability > 0.6:
        risk_category = 'High Risk'
    elif risk_probability > 0.3:
        risk_category = 'Moderate Risk'
    else:
        risk_category = 'Low Risk'

    # 6. Return the result as JSON
    response = {
        'patient_id': json_data.get('patient_id', 'Unknown'),
        'risk_probability': float(risk_probability),
        'risk_category': risk_category
    }
    
    return jsonify(response)

# ----------------------------------------------------------------------
# 5. Run the App
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 'host=0.0.0.0' makes it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)
