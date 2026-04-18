import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "yolo11n.pt"


def classify_team(player_crop):
    hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    hsv = hsv[: max(height // 2, 1), :]

    avg_val = np.mean(hsv[:, :, 2])
    if avg_val < 80:
        return "other"

    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([165, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    white_lower = np.array([0, 0, 160])
    white_upper = np.array([180, 100, 255])

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, red_lower1, red_upper1),
        cv2.inRange(hsv, red_lower2, red_upper2),
    )
    mask_white = cv2.inRange(hsv, white_lower, white_upper)

    red_ratio = np.sum(mask_red > 0) / (height * width)
    white_ratio = np.sum(mask_white > 0) / (height * width)

    if red_ratio > 0.02 and red_ratio >= white_ratio:
        return "red"
    if white_ratio > 0.02 and white_ratio > red_ratio:
        return "white"
    return "other"


def annotate_frame(model, frame, tracker: str):
    results = model.track(frame, persist=True, tracker=tracker, verbose=False)
    red_count, white_count = 0, 0

    for box in results[0].boxes:
        if int(box.cls) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            continue

        team = classify_team(player_crop)
        if team == "red":
            red_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif team == "white":
            white_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.putText(frame, f"Red team: {red_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"White team: {white_count}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def process_video(video_path: Path, model_path: Path, tracker: str, output_video: Path | None, show: bool) -> None:
    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    writer = None
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

    try:
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break

            annotated = annotate_frame(model=model, frame=frame, tracker=tracker)
            if writer is not None:
                writer.write(annotated)

            if show:
                cv2.imshow("YOLO11 Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track players and estimate team counts from a sports video.")
    parser.add_argument("--video-path", type=Path, required=True, help="Input video file.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="YOLO model weights.")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config passed to Ultralytics.")
    parser.add_argument("--output-video", type=Path, default=None, help="Optional path for an annotated output video.")
    parser.add_argument("--show", action="store_true", help="Display the annotated frames in a window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_video(
        video_path=args.video_path,
        model_path=args.model_path,
        tracker=args.tracker,
        output_video=args.output_video,
        show=args.show,
    )


if __name__ == "__main__":
    main()
