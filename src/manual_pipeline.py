import os
import time
import urllib.request

import cv2
import mediapipe as mp


HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
CAMERA_INDEX = 0
MAX_NUM_HANDS = 2
MAX_NUM_FACES = 10
FACE_BOX_MARGIN_RATIO = 0.2
FACE_BLUR_KERNEL = (99, 99)

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
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


# def ensure_model_exists(model_path, model_url):
#     if os.path.exists(model_path):
#         return

#     print(f"Downloading model: {model_path}")
#     try:
#         urllib.request.urlretrieve(model_url, model_path)
#     except Exception as exc:
#         raise RuntimeError(f"Failed to download model from {model_url}") from exc

#     if not os.path.exists(model_path):
#         raise RuntimeError(f"Model was not created: {model_path}")


def create_face_detector():
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_detection_confidence=0.5,
        min_suppression_threshold=0.3,
    )
    return FaceDetector.create_from_options(options)


def create_hand_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def detection_to_box(detection):
    if not detection.bounding_box:
        return None
    bbox = detection.bounding_box
    x1 = int(bbox.origin_x)
    y1 = int(bbox.origin_y)
    x2 = int(bbox.origin_x + bbox.width)
    y2 = int(bbox.origin_y + bbox.height)
    return x1, y1, x2, y2


def expand_and_clip_box(box, image_w, image_h, margin_ratio):
    x1, y1, x2, y2 = box
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)

    margin_x = int(box_w * margin_ratio)
    margin_y = int(box_h * margin_ratio)

    ex1 = max(0, x1 - margin_x)
    ey1 = max(0, y1 - margin_y)
    ex2 = min(image_w, x2 + margin_x)
    ey2 = min(image_h, y2 + margin_y)

    if ex2 <= ex1 or ey2 <= ey1:
        return None
    return ex1, ey1, ex2, ey2


def blur_face_region(frame, box):
    x1, y1, x2, y2 = box
    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return
    blurred = cv2.GaussianBlur(face_roi, FACE_BLUR_KERNEL, 0)
    frame[y1:y2, x1:x2] = blurred


def apply_face_blur(frame, detections):
    image_h, image_w = frame.shape[:2]
    for detection in detections:
        box = detection_to_box(detection)
        if box is None:
            continue
        safe_box = expand_and_clip_box(box, image_w, image_h, FACE_BOX_MARGIN_RATIO)
        if safe_box is None:
            continue
        blur_face_region(frame, safe_box)


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


def draw_fingertip_markers(frame, landmarks_px):
    for finger_name, tip_idx in TIP_IDS.items():
        x_tip, y_tip = landmarks_px[tip_idx]
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


def to_pixel_landmarks(hand_landmarks, frame_w, frame_h):
    points = []
    for lm in hand_landmarks:
        x_px = int(lm.x * frame_w)
        y_px = int(lm.y * frame_h)
        points.append((x_px, y_px))
    return points


def main():
    # ensure_model_exists(HAND_MODEL_PATH, HAND_MODEL_URL)
    # ensure_model_exists(FACE_MODEL_PATH, FACE_MODEL_URL)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera. Try CAMERA_INDEX = 1 if 0 does not work."
        )

    prev_time = time.time()

    with create_face_detector() as face_detector, create_hand_landmarker() as hand_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)

            try:
                face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
                detections = face_result.detections if face_result and face_result.detections else []
            except Exception:
                detections = []

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            display_frame = frame.copy()
            apply_face_blur(display_frame, detections)

            if hand_result.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    handedness_label = "Unknown"
                    if hand_result.handedness and hand_idx < len(hand_result.handedness):
                        if hand_result.handedness[hand_idx]:
                            handedness_label = hand_result.handedness[hand_idx][0].category_name

                    landmarks_px = to_pixel_landmarks(hand_landmarks, w, h)
                    draw_hand_landmarks(display_frame, landmarks_px)

                    finger_count, fingers_up = count_raised_fingers(
                        hand_landmarks,
                        handedness_label,
                    )
                    draw_finger_labels(display_frame, landmarks_px, fingers_up)
                    draw_fingertip_markers(display_frame, landmarks_px)

                    wrist_x, wrist_y = landmarks_px[0]
                    cv2.putText(
                        display_frame,
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
                display_frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Finger Tracker", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

