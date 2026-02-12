# src/model.py
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def train_and_save_model():
    """
    Train a simple Random Forest classifier on Iris dataset
    and save the model to ../models/model.pkl
    """
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save model
    model_path = '../models/model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    # Train and save model when script is run directly
    train_and_save_model()