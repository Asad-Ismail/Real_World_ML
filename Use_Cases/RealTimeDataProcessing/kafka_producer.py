import time
import pandas as pd
from kafka import KafkaProducer
import logging
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## Produce
producer = KafkaProducer(bootstrap_servers='localhost:9092')


X_val = pd.read_csv('creditcardfraud/X_val.csv')


# Function to send data to Kafka
def send_to_kafka(row):
    try:
        # Convert row to string or JSON
        message =row.to_json().encode()
        # Send to Kafka topic
        producer.send('fraud-message', value=message)
        producer.flush()
        logging.info(f"Sent: {message}")
    except Exception as e:
        print(f"Error: {e}")

# Simulate streaming
for index, row in X_val.iterrows():
    send_to_kafka(row)
    time.sleep(1)  # Sleep for 1 second to simulate streaming
