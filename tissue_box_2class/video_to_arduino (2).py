
import time
from pathlib import Path
from collections import defaultdict

import cv2
import serial
from ultralytics import YOLO

# =========================================
# 1. Settings
# =========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(r"D:\AI(Dr.Ali trainning)\Graduation Project\DEMO_3\tissue_box_2class\tissue_box_2class\best.pt")

VIDEO_PATH = Path(r"D:\AI(Dr.Ali trainning)\Graduation Project\DEMO_3\tissue_box_2class\tissue_box_2class\Video\tissue production line.mp4")

ARDUINO_PORT = "COM7"
BAUD_RATE = 9600

CONFIDENCE = 0.5
VOTES_REQUIRED = 5

GOOD_LABEL = "good"
DEFECT_LABEL = "defective"


# =========================================
# 2. Connect to Arduino
# =========================================

arduino = None
cap = None

def send_control_signal(signal):
    """Send a command from the laptop to Arduino."""
    arduino.write((signal + "\n").encode("ascii"))
    arduino.flush()
    print(f"[CONTROL SIGNAL] Sent: {signal}")


try:
    # Check files
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    # Load YOLO model
    print("[INFO] Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    names = model.names
    if isinstance(names, dict):
        class_names = {
            int(k): str(v).strip().lower()
            for k, v in names.items()
        }
    else:
        class_names = {
            i: str(v).strip().lower()
            for i, v in enumerate(names)
        }

    print("[INFO] Classes:", class_names)

    good_ids = [
        k for k, v in class_names.items()
        if v == GOOD_LABEL
    ]
    defect_ids = [
        k for k, v in class_names.items()
        if v == DEFECT_LABEL
    ]

    if not good_ids or not defect_ids:
        raise ValueError(
            "Expected classes 'good' and 'defective'. "
            f"Found: {class_names}"
        )

    good_id = good_ids[0]
    defect_id = defect_ids[0]

    # Connect to Arduino
    print(f"[INFO] Connecting to Arduino on {ARDUINO_PORT}...")
    arduino = serial.Serial(
        ARDUINO_PORT,
        BAUD_RATE,
        timeout=1,
        write_timeout=1
    )

    # Arduino may reset when serial opens
    time.sleep(2)
    arduino.reset_input_buffer()

    print("[OK] Arduino connected.")

    # Open video
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open the video.")

    # votes per tracking ID
    votes = defaultdict(list)
    processed_ids = set()

    pass_count = 0
    reject_count = 0

    print("[INFO] Inspection started. Press Q to stop.")

    # =========================================
    # 3. Inference + Tracking + Control
    # =========================================

    while True:
        ok, frame = cap.read()

        if not ok:
            print("[INFO] End of video.")
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        if (
            boxes is not None
            and boxes.id is not None
            and boxes.cls is not None
        ):
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()

            for track_id, class_id in zip(ids, classes):

                if track_id in processed_ids:
                    continue

                if class_id not in (good_id, defect_id):
                    continue

                # True means defective
                votes[track_id].append(class_id == defect_id)

                if len(votes[track_id]) < VOTES_REQUIRED:
                    continue

                # Majority decision for this tracked product
                defective_votes = sum(votes[track_id])
                is_defective = (
                    defective_votes > VOTES_REQUIRED / 2
                )

                processed_ids.add(track_id)

                if is_defective:
                    reject_count += 1
                    print(
                        f"[DECISION] ID {track_id}: DEFECTIVE"
                    )

                    # Send control command to Arduino
                    send_control_signal("R")

                else:
                    pass_count += 1
                    print(
                        f"[DECISION] ID {track_id}: GOOD - PASS"
                    )

                    # Good product: do not activate relay

                print(
                    f"[COUNT] PASS={pass_count}, "
                    f"REJECT={reject_count}"
                )

        # Display detections
        output = result.plot()

        cv2.imshow("YOLO Inspection", output)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    print("\n[RESULT] Inspection finished.")
    print(f"Good products: {pass_count}")
    print(f"Defective products: {reject_count}")

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.")

except Exception as exc:
    print(f"\n[ERROR] {exc}")

finally:
    # Request relay OFF before closing
    if arduino is not None and arduino.is_open:
        try:
            send_control_signal("X")
            time.sleep(0.1)
        except Exception as exc:
            print(f"[WARNING] Could not send OFF command: {exc}")
        finally:
            arduino.close()

    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()
    print("[INFO] Program ended.")
