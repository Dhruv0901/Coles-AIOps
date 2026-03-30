import os
import time
import urllib.request
from collections import deque, defaultdict

import cv2
import mediapipe as mp


MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CAMERA_INDEX = 0
MAX_NUM_HANDS = 2
TRAIL_LENGTH = 20

TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

PIP_IDS = {
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def count_raised_fingers(landmarks, handedness_label):
    fingers_up = []

    thumb_tip = landmarks[TIP_IDS["thumb"]]
    thumb_ip = landmarks[3]

    if handedness_label == "Right":
        thumb_up = thumb_tip.x < thumb_ip.x
    else:
        thumb_up = thumb_tip.x > thumb_ip.x

    fingers_up.append(("thumb", thumb_up))

    for finger_name in ["index", "middle", "ring", "pinky"]:
        tip = landmarks[TIP_IDS[finger_name]]
        pip = landmarks[PIP_IDS[finger_name]]
        is_up = tip.y < pip.y
        fingers_up.append((finger_name, is_up))

    count = sum(1 for _, is_up in fingers_up if is_up)
    return count, fingers_up


def draw_finger_labels(frame, landmarks_px, fingers_up):
    for finger_name, is_up in fingers_up:
        tip_idx = TIP_IDS[finger_name]
        x, y = landmarks_px[tip_idx]
        label = f"{finger_name}: {'UP' if is_up else 'DOWN'}"
        cv2.putText(
            frame,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if is_up else (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


def draw_hand_landmarks(frame, landmarks_px):
    for start_idx, end_idx in HAND_CONNECTIONS:
        x1, y1 = landmarks_px[start_idx]
        x2, y2 = landmarks_px[end_idx]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for idx, (x, y) in enumerate(landmarks_px):
        cv2.circle(frame, (x, y), 4, (255, 0, 255), -1)
        cv2.putText(
            frame,
            str(idx),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera. Try CAMERA_INDEX = 1 if 0 does not work."
        )

    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    prev_time = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                    handedness_label = "Unknown"
                    if result.handedness and hand_idx < len(result.handedness):
                        if result.handedness[hand_idx]:
                            handedness_label = result.handedness[hand_idx][0].category_name

                    landmarks_px = []
                    for lm in hand_landmarks:
                        x_px = int(lm.x * w)
                        y_px = int(lm.y * h)
                        landmarks_px.append((x_px, y_px))

                    draw_hand_landmarks(frame, landmarks_px)

                    finger_count, fingers_up = count_raised_fingers(
                        hand_landmarks,
                        handedness_label
                    )
                    draw_finger_labels(frame, landmarks_px, fingers_up)

                    for finger_name, tip_idx in TIP_IDS.items():
                        x_tip, y_tip = landmarks_px[tip_idx]
                        trails[(hand_idx, finger_name)].append((x_tip, y_tip))

                        cv2.circle(frame, (x_tip, y_tip), 8, (255, 0, 255), -1)
                        cv2.putText(
                            frame,
                            finger_name,
                            (x_tip + 8, y_tip + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                        points = list(trails[(hand_idx, finger_name)])
                        for i in range(1, len(points)):
                            cv2.line(frame, points[i - 1], points[i], (255, 255, 0), 2)

                    wrist_x, wrist_y = landmarks_px[0]
                    cv2.putText(
                        frame,
                        f"{handedness_label} hand | Fingers up: {finger_count}",
                        (max(wrist_x - 20, 10), max(wrist_y - 20, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Finger Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()