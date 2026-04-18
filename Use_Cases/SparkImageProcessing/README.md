# Spark Image Processing

This folder contains two versions of the same task:

- `convert_to_gray_opencv.py`: simple local baseline with OpenCV
- `convert_to_gray_pyspark.py`: distributed-style version with PySpark

Both scripts now use relative paths and command-line arguments instead of hardcoded machine-specific paths.

## Install

```bash
pip install opencv-python pyspark numpy
```

## Optional Spark Environment

If you use a custom Python environment for Spark, set:

```bash
export PYSPARK_PYTHON="/path/to/python"
export PYSPARK_DRIVER_PYTHON="/path/to/python"
```

## Input Folder

Put a few `.jpg`, `.jpeg`, `.png`, or `.bmp` files into:

- `Use_Cases/SparkImageProcessing/input_images/`

or pass a custom directory with `--input-dir`.

## Run The OpenCV Baseline

```bash
python Use_Cases/SparkImageProcessing/convert_to_gray_opencv.py
```

Outputs are written to:

- `Use_Cases/SparkImageProcessing/opencv_gray_output/`

## Run The PySpark Version

```bash
python Use_Cases/SparkImageProcessing/convert_to_gray_pyspark.py --master "local[*]"
```

Outputs are written to:

- `Use_Cases/SparkImageProcessing/spark_gray_output/`

## Why Keep Both

The OpenCV script is the easiest first step for understanding the image transformation itself.
The PySpark script shows how the same transformation can be applied across a directory with a distributed data-processing pattern.
