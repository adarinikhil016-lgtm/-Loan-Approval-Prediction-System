"""
preprocessing.py
Data cleaning, missing value handling, categorical encoding,
feature scaling, and reusable preprocessing pipeline.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
class LoanDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessing pipeline for loan applicant data.
    Performs data validation, missing value imputation, feature engineering,
    one-hot encoding, and feature scaling.
    """
    def __init__(self):
        self.numeric_features = [
            "Age", "MonthlyIncome", "LoanAmount", "CreditScore",
            "ExistingLoans", "LoanTerm", "EMIToIncomeRatio", "LoanToIncomeRatio"
        ]
        self.categorical_features = ["Gender", "EmploymentStatus", "PropertyArea"]
        self.pipeline = None
        self.feature_names_out_ = None
        self.imputer_num_ = None
        self.imputer_cat_ = None
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific financial risk ratios."""
        df_copy = df.copy()
        
        # Ensure numeric types
        for col in ["Age", "MonthlyIncome", "LoanAmount", "CreditScore", "ExistingLoans", "LoanTerm"]:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
        # Handle missing income or loan term before ratio calculation
