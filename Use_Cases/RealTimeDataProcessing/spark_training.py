import argparse
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES_CSV = SCRIPT_DIR / "creditcardfraud" / "X_train.csv"
DEFAULT_LABELS_CSV = SCRIPT_DIR / "creditcardfraud" / "y_train.csv"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "trained_model"


def create_spark_session(app_name: str, master: str | None = None) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    return builder.getOrCreate()


def load_training_dataframe(
    spark: SparkSession,
    features_path: Path,
    labels_path: Path,
):
    df_features = spark.read.csv(str(features_path), header=True, inferSchema=True)
    df_labels = spark.read.csv(str(labels_path), header=True, inferSchema=True)

    df_features = df_features.withColumn("row_id", monotonically_increasing_id())
    df_labels = df_labels.withColumn("row_id", monotonically_increasing_id())
    return df_features.join(df_labels, "row_id").drop("row_id")


def train_model(
    features_csv: Path,
    labels_csv: Path,
    model_dir: Path,
    app_name: str,
    master: str | None = None,
) -> Path:
    spark = create_spark_session(app_name=app_name, master=master)
    try:
        training_df = load_training_dataframe(spark, features_csv, labels_csv)
        feature_columns = [column for column in training_df.columns if column != "Class"]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        classifier = RandomForestClassifier(labelCol="Class", featuresCol="features")
        pipeline = Pipeline(stages=[assembler, classifier])

        print("Training Spark fraud model...")
        model = pipeline.fit(training_df)
        model.write().overwrite().save(str(model_dir))
        print(f"Saved model to {model_dir}")
        return model_dir
    finally:
        spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Spark RandomForest fraud model from CSV splits."
    )
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--app-name",
        type=str,
        default="FraudDetectionModelTraining",
        help="Spark application name.",
    )
    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help="Optional Spark master, for example 'local[*]'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        features_csv=args.features_csv,
        labels_csv=args.labels_csv,
        model_dir=args.model_dir,
        app_name=args.app_name,
        master=args.master,
    )


if __name__ == "__main__":
    main()
