import os
import string
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt_tab')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transform text by removing punctuation, stopwords, and applying stemming."""
    try:
        text = text.lower()
        text = nltk.word_tokenize(text)
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text if word not in stopwords.words('english') and word not in string.punctuation]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error transforming text: {e}")
        raise

def preprocess_df(df,text_column, target_column):
    """Preprocess the DataFrame by transforming text and encoding labels."""
    try:
        df[text_column] = df[text_column].apply(transform_text)
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        logger.info("Target column encoded successfully.")

        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        logger.info("Duplicate rows deleted successfully.")

        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.info("Text column transformed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing DataFrame: {e}")
        raise

def main():
    try:
        train_df = pd.read_csv('./data/raw/train.csv')
        test_df = pd.read_csv('./data/raw/test.csv')
        logger.info("Data loaded successfully.")

        processed_train_df = preprocess_df(train_df, text_column='text', target_column='target')
        processed_test_df = preprocess_df(test_df, text_column='text', target_column='target')

        data_path = os.path.join("./data","interim")
        os.makedirs(data_path, exist_ok=True)

        processed_train_df.to_csv(os.path.join(data_path, 'processed_train.csv'), index=False)
        processed_test_df.to_csv(os.path.join(data_path, 'processed_test.csv'), index=False)
        logger.info(f"Processed data has been saved successfully to {data_path}")

    except Exception as e:
        logger.error(f"Error in data transformation: {e}")
        raise

if __name__ == "__main__":
    main()