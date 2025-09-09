import os
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
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
        data.fillna('', inplace=True)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def apply_tfidf(train_data, test_data, max_f):
    """Apply TF-IDF vectorization to text data."""
    try:
        tfidf = TfidfVectorizer(max_features=max_f)
        X_train_tfidf = tfidf.fit_transform(train_data['text'].values)
        X_test_tfidf = tfidf.transform(test_data['text'].values)
        
        
        X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray())
        X_train_tfidf_df['label'] = train_data['target'].values
        X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray())
        X_test_tfidf_df['label'] = test_data['target'].values
        
        logger.info("TF-IDF vectorization applied successfully.")
        return X_train_tfidf_df, X_test_tfidf_df
    except Exception as e:
        logger.error(f"Error applying TF-IDF vectorization: {e}")
        raise

def save_data(df, file_path):
    """Save DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def main():
    try:
        max_features = 50

        train_data = load_data('./data/interim/processed_train.csv')
        test_data = load_data('./data/interim/processed_test.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, './data/processed/train_tfidf.csv')
        save_data(test_df, './data/processed/test_tfidf.csv')

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()