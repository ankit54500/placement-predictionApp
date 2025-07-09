from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Global variables to store model and columns
model = None
expected_columns = None

def load_model():
    """Load the trained model and expected columns from joblib files"""
    global model, expected_columns
    try:
        # Load your model
        model = joblib.load('placement_model.pkl')
        print("Model loaded successfully!")
        
        # Load expected columns
        try:
            expected_columns = joblib.load('model_columns.pkl')
            print(f"Expected columns loaded: {expected_columns}")
        except FileNotFoundError:
            print("model_columns.pkl not found. Using default columns.")
            expected_columns = ['cgpa', 'resume_score']
            
        return True
    except FileNotFoundError:
        print("Model file 'placement_model.pkl' not found. Please ensure it exists.")
        return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on user input"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features
        cgpa = float(data.get('cgpa', 0))
        resume_score = float(data.get('resume_score', 0))
        
        # Validate input
        if cgpa < 0 or cgpa > 10:
            return jsonify({'error': 'CGPA must be between 0 and 10'}), 400
        
        if resume_score < 0 or resume_score > 100:
            return jsonify({'error': 'Resume score must be between 0 and 100'}), 400
        
        # Prepare input for prediction using expected columns order
        if expected_columns:
            # Create input array in the expected order
            input_features = np.array([[cgpa, resume_score]])
        else:
            # Fallback to default order
            input_features = np.array([[cgpa, resume_score]])
        
        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        prediction = model.predict(input_features)
        prediction_proba = None
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(input_features)
            confidence = float(np.max(prediction_proba))
        else:
            confidence = None
        
        # Interpret prediction
        placement_status = "Placed" if prediction[0] == 1 else "Not Placed"
        
        # Prepare response
        response = {
            'prediction': placement_status,
            'prediction_value': int(prediction[0]),
            'cgpa': cgpa,
            'resume_score': resume_score,
            'confidence': confidence
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': 'Invalid input format'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'model_type': type(model).__name__,
        'features': expected_columns if expected_columns else ['CGPA', 'Resume Score'],
        'target': 'Placement Status (0: Not Placed, 1: Placed)',
        'expected_columns': expected_columns
    }
    
    return jsonify(info)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# ✅ Load model on startup (even with gunicorn)
model_loaded = load_model()

if not model_loaded:
    print("Warning: Model not loaded. Please check your pickle file.")
