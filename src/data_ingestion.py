import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
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

def preprocess_data(data):
    """Preprocess the data (e.g., handle missing values)."""
    try:
        data.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
        data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.info("Data preprocessing completed successfully.")
        return data
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise 

def save_data(train_data,test_data,file_path):
    """Save data to a CSV file."""
    try:
        raw_data_path = os.path.join(file_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        ts = params['data_ingestion']['test_size']
        data_path = 'https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/refs/heads/main/experiments/spam.csv'
        df = load_data(data_path)
        processed_df = preprocess_data(df)
        train_data, test_data = train_test_split(processed_df, test_size=ts)
        logger.info(f"Train-test split completed: {len(train_data)} train, {len(test_data)} test samples.")
        save_data(train_data, test_data, './data')
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise

if __name__ == "__main__":
    main()


