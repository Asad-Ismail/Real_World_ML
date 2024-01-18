import time
import pandas as pd
from kafka import KafkaProducer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## Produce
producer = KafkaProducer(bootstrap_servers='localhost:9092')


# Read your CSV file
df = pd.read_csv('your_data.csv')

# Function to send data to Kafka
def send_to_kafka(row):
    try:
        # Convert row to string or JSON
        message = row.to_json().encode()
        # Send to Kafka topic, e.g., 'test-topic'
        producer.send('test-topic', value=message)
        producer.flush()
        print(f"Sent: {message}")
    except Exception as e:
        print(f"Error: {e}")

# Simulate streaming
for index, row in df.iterrows():
    send_to_kafka(row)
    time.sleep(1)  # Sleep for 1 second to simulate streaming
