import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "creditcardfraud" / "X_val.csv"


def row_to_payload(row: pd.Series) -> dict[str, float]:
    return {column: float(value) for column, value in row.to_dict().items()}


def stream_rows(
    input_csv: Path,
    bootstrap_servers: str,
    topic: str,
    delay_seconds: float,
    max_messages: int | None = None,
) -> int:
    dataframe = pd.read_csv(input_csv)
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda message: json.dumps(message).encode("utf-8"),
    )

    sent = 0
    try:
        for _, row in dataframe.iterrows():
            payload = row_to_payload(row)
            producer.send(topic, value=payload).get(timeout=10)
            sent += 1
            logging.info("Sent message %s to topic '%s'", sent, topic)
            if max_messages is not None and sent >= max_messages:
                break
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    finally:
        producer.flush()
        producer.close()
    return sent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream validation rows from the fraud dataset into a Kafka topic."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="CSV file containing the feature rows to stream.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:9092",
        help="Kafka bootstrap servers.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="fraud-message",
        help="Kafka topic name.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between streamed rows.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Optional cap for smoke-testing the producer.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    total = stream_rows(
        input_csv=args.input_csv,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        delay_seconds=args.delay_seconds,
        max_messages=args.max_messages,
    )
    logging.info("Finished streaming %s rows from %s", total, args.input_csv)


if __name__ == "__main__":
    main()
