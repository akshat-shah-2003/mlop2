import os
import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def train_model(X_train,y_train,params):
    """Train a RandomForest model."""
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train do not match.")
        
        logger.info(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
        logger.info(f"Training parameters: {params}")
        model = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        logger.info(f"Starting model training with {X_train.shape[0]} samples...")
        model.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def save_model(model, file_path):
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise

def main():
    try:
        params = {'n_estimators': 25,'random_state': 2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        model = train_model(X_train,y_train,params)
        save_model(model, './models/rf_model.pkl')
    
    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise

if __name__ == "__main__":
    main()