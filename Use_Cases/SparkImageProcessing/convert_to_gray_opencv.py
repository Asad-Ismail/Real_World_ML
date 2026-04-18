import argparse
import time
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input_images"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "opencv_gray_output"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_image_paths(input_dir: Path):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def convert_directory(input_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    for image_path in iter_image_paths(input_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable file: {image_path}")
            continue
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), gray_image)
        processed_count += 1

    return processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a directory of images to grayscale with OpenCV."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.monotonic()
    processed_count = convert_directory(args.input_dir, args.output_dir)
    duration = time.monotonic() - start_time
    print(f"Processed {processed_count} images in {duration:.3f} seconds")


if __name__ == "__main__":
    main()
