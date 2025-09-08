import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO11 model
model = YOLO("yolo11n.pt")

# Open video file
video_path = "/home/asad/Downloads/sample.mp4"
cap = cv2.VideoCapture(video_path)

def classify_team(player_crop):
    hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    hsv = hsv[0:h//2, :]

    # Check brightness (ignore dark regions = spectators)
    avg_val = np.mean(hsv[:,:,2])
    if avg_val < 80:
        return "other"

    # Relaxed HSV ranges
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([165, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    white_lower = np.array([0, 0, 160])
    white_upper = np.array([180, 100, 255])

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, red_lower1, red_upper1),
        cv2.inRange(hsv, red_lower2, red_upper2)
    )
    mask_white = cv2.inRange(hsv, white_lower, white_upper)

    red_ratio = np.sum(mask_red > 0) / (h * w)
    white_ratio = np.sum(mask_white > 0) / (h * w)

    min_ratio = 0.02

    if red_ratio > min_ratio and red_ratio >= white_ratio:
        return "red"
    elif white_ratio > min_ratio and white_ratio > red_ratio:
        return "white"
    else:
        return "other"


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    red_count, white_count = 0, 0

    for box in results[0].boxes:
        if int(box.cls) != 0:  # Only keep 'person' class
            continue

        # Get bounding box coords
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        player_crop = frame[y1:y2, x1:x2]

        if player_crop.size == 0:
            continue

        # Classify team
        team = classify_team(player_crop)

        if team == "red":
            red_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
        elif team == "white":
            white_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
        # spectators/referees (team=="other") are skipped
        
    # Annotate frame with counts
    cv2.putText(frame, f"Red team: {red_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"White team: {white_count}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Show frame
    cv2.imshow("YOLO11 Tracking", frame)

    cv2.waitKey()
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
