import argparse
import multiprocessing
import os
import time
from pathlib import Path

import cv2
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import BinaryType


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input_images"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "spark_gray_output"


@udf(BinaryType())
def process_image(binary: bytes) -> bytes:
    image_array = np.asarray(bytearray(binary), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return b""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    success, encoded_image = cv2.imencode(".jpg", gray_image)
    if not success:
        return b""
    return encoded_image.tobytes()


def create_spark_session(master: str, driver_memory: str) -> SparkSession:
    return (
        SparkSession.builder.master(master)
        .config("spark.driver.memory", driver_memory)
        .appName("SparkImageProcessing")
        .getOrCreate()
    )


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    master: str = "local[*]",
    driver_memory: str = "2g",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spark = create_spark_session(master=master, driver_memory=driver_memory)
    spark.sparkContext.setLogLevel("WARN")
    output_dir_broadcast = spark.sparkContext.broadcast(str(output_dir))

    try:
        num_cores = multiprocessing.cpu_count()
        print(f"Number of logical cores available: {num_cores}")

        def write_images_to_disk(partition):
            local_output_dir = Path(output_dir_broadcast.value)
            local_output_dir.mkdir(parents=True, exist_ok=True)
            rows_processed = 0
            for row in partition:
                processed_image = row["processed_image"]
                if not processed_image:
                    continue
                output_filename = local_output_dir / Path(row["path"]).name
                with open(output_filename, "wb") as file:
                    file.write(processed_image)
                rows_processed += 1
            print(f"Rows processed in this partition: {rows_processed}")

        start_time = time.monotonic()
        df = spark.read.format("binaryFile").load(str(input_dir))
        df_processed = df.select("path", process_image(col("content")).alias("processed_image"))
        df_processed.foreachPartition(write_images_to_disk)
        duration = time.monotonic() - start_time
        print(f"Processed {df_processed.count()} files in {duration:.3f} seconds")
    finally:
        output_dir_broadcast.unpersist()
        spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a directory of images to grayscale with PySpark."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--master", type=str, default="local[*]")
    parser.add_argument("--driver-memory", type=str, default="2g")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        master=args.master,
        driver_memory=args.driver_memory,
    )


if __name__ == "__main__":
    main()
