import os
import logging
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path='params.yaml'):
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Parameters loaded successfully from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters from {params_path}: {e}")
        raise

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def load_model(file_path):
    """Load a trained model from a file."""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    try:        
        logger.info(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
        logger.info(f"Evaluating model with {X_test.shape[0]} samples...")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
        
        logger.info(f"Evaluation metrics calculated as: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def save_metrics(metrics, file_path):
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {file_path}: {e}")
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        model_path = './models/rf_model.pkl'
        test_data_path = './data/processed/test_tfidf.csv'
        metrics_output_path = './reports/metrics.json'

        model = load_model(model_path)
        test_data = load_data(test_data_path)

        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        metrics = evaluate_model(model, X_test, y_test)

        with Live(save_dvc_exp=True) as live:
            for i,j in metrics.items():
                live.log_metric(i, j)
            live.log_params(params)
        save_metrics(metrics, metrics_output_path)
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise
if __name__ == "__main__":
    main()



