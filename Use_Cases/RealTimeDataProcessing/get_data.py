import logging
import requests
import zipfile
import io
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_data():
    """
    Downloads and extracts the dataset from a specified URL.
    """
    url = "https://s3-us-west-2.amazonaws.com/sagemaker-e2e-solutions/fraud-detection/creditcardfraud.zip"
    try:
        logging.info("Starting to download the file...")
        response = requests.get(url)
        zip_content = response.content
        logging.info("Download complete. Extracting files...")

        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            zip_ref.extractall("creditcardfraud")
        logging.info("Successfully extracted files")
    except requests.RequestException as e:
        logging.error(f"Error during downloading: {e}")
    except zipfile.BadZipFile as e:
        logging.error(f"Error during extraction: {e}")

def load_data():
    """
    Loads the dataset, either by downloading it or reading a local file.
    Splits the data into training and validation sets.

    Returns:
        tuple: A tuple containing the training and validation data.
    """
    if not os.path.exists('creditcardfraud/creditcard.csv'):
        get_data()

    model_data = pd.read_csv('creditcardfraud/creditcard.csv', delimiter=',')
    model_data.reset_index(inplace=True)

    X = model_data.drop('Class', axis=1).values
    y = model_data['Class'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    logging.info(f"Data loaded with training shape: {X_train.shape}")

    # Saving to CSV files
    np.savetxt('creditcardfraud/X_train.csv', X_train, delimiter=',')
    np.savetxt('creditcardfraud/X_val.csv', X_val, delimiter=',')
    np.savetxt('creditcardfraud/y_train.csv', y_train, delimiter=',')
    np.savetxt('creditcardfraud/y_val.csv', y_val, delimiter=',')


if __name__=="__main__":
    load_data()