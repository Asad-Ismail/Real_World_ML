from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import BinaryType
import numpy as np
import cv2
import multiprocessing
import time

import os

# Corrected paths
img_path = "/Users/gmeax/Downloads/example_images"
out_path = "/Users/gmeax/Downloads/example_process_images"  # Removed the trailing space

# Initialize SparkSession
#spark = SparkSession.builder.appName("ImageProcessing").getOrCreate()
# local mode use as many threads as cores
spark = SparkSession.builder.master("local[*]").appName("ImageProcessing").getOrCreate()

# Log the number of logical cores available on your machine
num_cores = multiprocessing.cpu_count()
print(f"Number of logical cores available: {num_cores}")


# Define UDF for processing image
@udf(BinaryType())
def process_image(binary):
    # Convert binary to NumPy array
    image = np.asarray(bytearray(binary), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Convert to CV2 image
    
    # Process the image (example: convert to grayscale)
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert back to binary
    _, encoded_image = cv2.imencode('.jpg', processed_image)
    return encoded_image.tobytes()


def write_images_to_disk(partition):
    rows_processed = 0
    for row in partition:
        rows_processed += 1
        processed_image = row['processed_image']
        output_filename = os.path.join(broadcast_out_path.value, os.path.basename(row['path']))
        # Write the processed image to disk
        with open(output_filename, 'wb') as file:
            file.write(processed_image)
    print(f"Rows processed in this partition: {rows_processed}")



# Load images as binary files
df = spark.read.format("binaryFile").load(img_path)


start_time = time.monotonic()

# Process images
df_processed = df.withColumn("processed_image", process_image(col("content")))


end_time = time.monotonic()
print(f"Execution time: {end_time - start_time} seconds")

# Broadcast the output path to all nodes (good practice for distributed environments)
broadcast_out_path = spark.sparkContext.broadcast(out_path)


# Use foreachPartition to distribute the image writing process
df_processed.foreachPartition(write_images_to_disk)

spark.stop()
