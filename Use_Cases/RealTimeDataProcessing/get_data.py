import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split


DATASET_URL = (
    "https://s3-us-west-2.amazonaws.com/"
    "sagemaker-e2e-solutions/fraud-detection/creditcardfraud.zip"
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "creditcardfraud"


def download_and_extract_dataset(output_dir: Path, url: str = DATASET_URL) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading fraud dataset to %s", output_dir)
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content), "r") as archive:
        archive.extractall(output_dir)

    csv_path = output_dir / "creditcard.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected dataset file not found at {csv_path}")
    logging.info("Dataset ready at %s", csv_path)
    return csv_path


def ensure_dataset(dataset_dir: Path) -> Path:
    csv_path = dataset_dir / "creditcard.csv"
    if csv_path.exists():
        return csv_path
    return download_and_extract_dataset(dataset_dir)


def prepare_splits(
    dataset_dir: Path,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, Path]:
    csv_path = ensure_dataset(dataset_dir)
    model_data = pd.read_csv(csv_path)
    X = model_data.drop("Class", axis=1)
    y = model_data["Class"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    output_paths = {
        "X_train": dataset_dir / "X_train.csv",
        "X_val": dataset_dir / "X_val.csv",
        "y_train": dataset_dir / "y_train.csv",
        "y_val": dataset_dir / "y_val.csv",
    }
    X_train.to_csv(output_paths["X_train"], index=False)
    X_val.to_csv(output_paths["X_val"], index=False)
    y_train.to_csv(output_paths["y_train"], index=False)
    y_val.to_csv(output_paths["y_val"], index=False)

    logging.info(
        "Prepared splits: train=%s, val=%s",
        X_train.shape,
        X_val.shape,
    )
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the credit-card fraud dataset and create train/validation CSV splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where the dataset and derived CSV files will be stored.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of rows used for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/validation split.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    paths = prepare_splits(
        dataset_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    for name, path in paths.items():
        logging.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
