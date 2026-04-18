# Real-Time Fraud Detection With Kafka + Spark

This folder shows a minimal end-to-end streaming workflow:

1. download and split a tabular fraud dataset
2. train a Spark model on the historical data
3. stream validation rows into Kafka
4. score those rows with Spark Structured Streaming

## What You Need

- Python dependencies from `requirements.txt`
- Apache Spark installed locally
- Kafka running locally on `localhost:9092`

If you want to run Spark locally, `local[*]` is a good default master value.

## Install

```bash
pip install -r Use_Cases/RealTimeDataProcessing/requirements.txt
```

## 1. Download And Split The Dataset

```bash
python Use_Cases/RealTimeDataProcessing/get_data.py
```

This creates:

- `Use_Cases/RealTimeDataProcessing/creditcardfraud/X_train.csv`
- `Use_Cases/RealTimeDataProcessing/creditcardfraud/X_val.csv`
- `Use_Cases/RealTimeDataProcessing/creditcardfraud/y_train.csv`
- `Use_Cases/RealTimeDataProcessing/creditcardfraud/y_val.csv`

## 2. Train The Spark Model

```bash
python Use_Cases/RealTimeDataProcessing/spark_training.py --master "local[*]"
```

This writes the trained Spark pipeline to:

- `Use_Cases/RealTimeDataProcessing/trained_model/`

## 3. Start Streaming Inference

Start Kafka first, then run:

```bash
python Use_Cases/RealTimeDataProcessing/spark_inference.py --master "local[*]"
```

The inference job reads JSON messages from the Kafka topic `fraud-message` and prints predictions to the console.

## 4. Publish Validation Rows Into Kafka

In a second terminal:

```bash
python Use_Cases/RealTimeDataProcessing/kafka_producer.py --delay-seconds 0.5 --max-messages 20
```

Use `--max-messages` for a quick smoke test. Omit it to stream the full validation CSV.

## Kafka Sanity Check

If you want to verify Kafka itself before running Spark, you can still use the CLI tools:

```bash
bin/kafka-console-producer.sh --topic fraud-message --bootstrap-server localhost:9092
bin/kafka-console-consumer.sh --topic fraud-message --from-beginning --bootstrap-server localhost:9092
```

## Notes

- All scripts are now safe to import without immediately starting training or streaming.
- The dataset download requires internet access.
- Spark Structured Streaming with Kafka requires the Kafka Spark package configured in `spark_inference.py`.
