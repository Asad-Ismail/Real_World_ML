import argparse
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import FloatType, StructField, StructType


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "trained_model"
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / "stream_checkpoints"


def create_feature_schema() -> StructType:
    fields = [StructField("Time", FloatType(), True)]
    fields.extend(StructField(f"V{i}", FloatType(), True) for i in range(1, 29))
    fields.append(StructField("Amount", FloatType(), True))
    return StructType(fields)


def create_spark_session(app_name: str, bootstrap_servers: str, master: str | None = None) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .config("spark.sql.shuffle.partitions", "2")
    )
    if master:
        builder = builder.master(master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def start_inference_stream(
    model_dir: Path,
    bootstrap_servers: str,
    topic: str,
    checkpoint_dir: Path,
    app_name: str,
    master: str | None = None,
    num_rows: int = 5,
):
    spark = create_spark_session(app_name=app_name, bootstrap_servers=bootstrap_servers, master=master)
    schema = create_feature_schema()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", bootstrap_servers)
            .option("subscribe", topic)
            .load()
        )

        parsed = (
            df.selectExpr("CAST(value AS STRING) AS json")
            .select(from_json(col("json"), schema).alias("data"))
            .select("data.*")
        )

        model = PipelineModel.load(str(model_dir))
        predictions = model.transform(parsed).select("Time", "Amount", "prediction", "probability")

        query = (
            predictions.writeStream.outputMode("append")
            .format("console")
            .option("truncate", False)
            .option("numRows", num_rows)
            .option("checkpointLocation", str(checkpoint_dir))
            .start()
        )
        query.awaitTermination()
    finally:
        spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Spark Structured Streaming inference over Kafka fraud messages."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092")
    parser.add_argument("--topic", type=str, default="fraud-message")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--app-name",
        type=str,
        default="FraudDetectionStreamingInference",
        help="Spark application name.",
    )
    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help="Optional Spark master, for example 'local[*]'.",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=5,
        help="Number of rows shown per console update.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_inference_stream(
        model_dir=args.model_dir,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        checkpoint_dir=args.checkpoint_dir,
        app_name=args.app_name,
        master=args.master,
        num_rows=args.num_rows,
    )


if __name__ == "__main__":
    main()
