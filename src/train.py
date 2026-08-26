"""
train.py
Model training, evaluation, comparison, visualization generation,
and model serialization for Loan Approval Prediction System.
"""
import os
import sys
# Ensure non-GUI backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
# Add src to sys.path if not present
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from preprocessing import LoanDataPreprocessor, clean_raw_data
from data_generator import generate_loan_data
# Set styling for plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300
def create_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Generate and save required and bonus EDA visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n[INFO] Generating Exploratory Data Visualizations...")
