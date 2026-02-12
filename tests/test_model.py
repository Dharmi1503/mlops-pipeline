# tests/test_model.py
import sys
import os
import joblib
import numpy as np
import pandas as pd
import pytest

# Get the absolute path to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

def test_model_loads_successfully():
    """Test 1: Verify model file loads successfully"""
    try:
        # CORRECT PATHS - matches your folder structure
        model_path = os.path.join(project_root, 'models', 'churn_model.pkl')
        columns_path = os.path.join(project_root, 'models', 'model_columns.pkl')
        
        print(f"🔍 Looking for model at: {model_path}")
        print(f"🔍 Looking for columns at: {columns_path}")
        
        # Check if files exist before loading
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        assert os.path.exists(columns_path), f"Columns file not found at {columns_path}"
        
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        
        assert model is not None
        assert model_columns is not None
        assert len(model_columns) == 12
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model type: {type(model).__name__}")
        print(f"🔢 Features expected: {len(model_columns)}")
        print(f"📋 First 5 features: {list(model_columns)[:5]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📁 Models directory exists: {os.path.exists(os.path.join(project_root, 'models'))}")
        if os.path.exists(os.path.join(project_root, 'models')):
            print(f"📁 Files in models: {os.listdir(os.path.join(project_root, 'models'))}")
        pytest.fail(f"❌ Model failed to load: {e}")

def test_prediction_output_shape():
    """Test 2: Verify prediction works with correct input shape"""
    # Load model and columns
    model_path = os.path.join(project_root, 'models', 'churn_model.pkl')
    columns_path = os.path.join(project_root, 'models', 'model_columns.pkl')
    
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    
    # Create a sample customer with ALL 12 features
    sample_customer = pd.DataFrame([[
        12,        # tenure: 12 months
        85.5,      # MonthlyCharges: $85.50
        0,         # SeniorCitizen: No
        0,         # Contract_One year: No
        1,         # Contract_Two year: Yes
        0,         # InternetService_Fiber optic: No
        0,         # InternetService_No: No
        0,         # OnlineSecurity_No internet service: No
        1,         # OnlineSecurity_Yes: Yes
        0,         # PaymentMethod_Credit card (automatic): No
        1,         # PaymentMethod_Electronic check: Yes
        0          # PaymentMethod_Mailed check: No
    ]], columns=model_columns)
    
    # Make prediction
    prediction = model.predict(sample_customer)
    prediction_proba = model.predict_proba(sample_customer)
    
    # Check shapes
    assert prediction.shape == (1,), f"Expected shape (1,), got {prediction.shape}"
    assert prediction_proba.shape == (1, 2), f"Expected shape (1,2), got {prediction_proba.shape}"
    
    print(f"✅ Prediction shape correct: {prediction.shape}")
    print(f"🔮 Prediction: {'Will Churn' if prediction[0] == 1 else 'Will Stay'}")
    print(f"📊 Probability: No Churn: {prediction_proba[0][0]:.3f}, Churn: {prediction_proba[0][1]:.3f}")

def test_multiple_customers():
    """Test 3: Test multiple customer predictions"""
    # Load model and columns
    model_path = os.path.join(project_root, 'models', 'churn_model.pkl')
    columns_path = os.path.join(project_root, 'models', 'model_columns.pkl')
    
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    
    # Test 3 different customer profiles
    customers = pd.DataFrame([
        [12, 85.5, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],  # Loyal customer
        [2, 120.5, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],  # New, expensive
        [70, 45.0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]   # Long-term, no internet
    ], columns=model_columns)
    
    # Make predictions
    predictions = model.predict(customers)
    
    # Check shape
    assert predictions.shape == (3,), f"Expected shape (3,), got {predictions.shape}"
    
    print(f"✅ Multiple predictions shape correct: {predictions.shape}")
    print("📋 Customer predictions:")
    for i, pred in enumerate(predictions):
        status = "⚠️ WILL CHURN" if pred == 1 else "✅ WILL STAY"
        print(f"   Customer {i+1}: {status}")

if __name__ == "__main__":
    print("🧪 Testing Churn Prediction Model...\n")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Models directory: {os.path.join(project_root, 'models')}")
    
    # Debug: List files in models directory
    models_dir = os.path.join(project_root, 'models')
    if os.path.exists(models_dir):
        print(f"📁 Files in models folder: {os.listdir(models_dir)}")
    else:
        print(f"❌ Models folder not found!")
    
    print("\n" + "="*50 + "\n")
    
    # Run tests
    test_model_loads_successfully()
    print("\n" + "="*50 + "\n")
    test_prediction_output_shape()
    print("\n" + "="*50 + "\n")
    test_multiple_customers()
    print("\n" + "="*50 + "\n")
    print("🎉 All tests passed! Your model is ready for deployment!")