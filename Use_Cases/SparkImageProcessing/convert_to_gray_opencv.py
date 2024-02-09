import numpy as np
import cv2
import time
import os

# Corrected paths
img_path = "/Users/gmeax/Downloads/example_images"
out_path = "/Users/gmeax/Downloads/example_process_images"  # Removed the trailing space

images=os.listdir(img_path)

for img_f in images:
    start_time = time.monotonic()
    img_f=os.path.join(img_path,img_f)
    img=cv2.imread(img_f)
    processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

end_time = time.monotonic()
print(f"Execution time: {end_time - start_time} seconds")
