"""
predict.py
Inference engine and credit risk explanation logic for loan approval prediction.
"""
import os
import joblib
import numpy as np
import pandas as pd
class LoanPredictor:
    """
    Loan application inference engine.
    Wraps the trained preprocessor and ML classification model.
    """
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(current_dir)
            models_dir = os.path.join(project_dir, "models")
        self.models_dir = models_dir
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        self.model_path = os.path.join(models_dir, "best_model.pkl")
        if not os.path.exists(self.preprocessor_path) or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifacts not found in {models_dir}. Please run 'python src/train.py' first."
            )
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
    def predict_single(self, applicant_data: dict) -> dict:
        """
        Predict loan approval for a single applicant record.
        Parameters:
        -----------
        applicant_data : dict
            Dictionary containing applicant features:
            - Age (int)
            - MonthlyIncome (float)
