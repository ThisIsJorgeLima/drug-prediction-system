# tests/test_drug_predictor.py
import pytest
import pandas as pd
import numpy as np
from src.models.drug_predictor import DrugPredictor

def test_drug_predictor():
    # Create sample data
    X = np.random.rand(100, 5)
    y = np.random.choice(['drugA', 'drugB', 'drugC'], size=100)

    # Initialize predictor
    predictor = DrugPredictor()

    # Test training
    accuracy = predictor.train(X, y)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

    # Test prediction
    input_data = np.random.rand(5)
    prediction = predictor.predict(input_data)
    assert isinstance(prediction[0], str)
    assert prediction[0] in ['drugA', 'drugB', 'drugC']
