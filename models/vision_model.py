import cv2
import numpy as np
import os
import time
import threading
from ultralytics import YOLO
from models.motion_detector import MotionDetector

MODEL_DIR = os.path.dirname(__file__)
PHONE_CAM_URL = "http://web_cam_IP:port/video"  # backup cam ip // use the cam ip if available

crib_obs_model = YOLO(os.path.join(MODEL_DIR, "crib_obs_model.tflite"))
posture_model = YOLO(os.path.join(MODEL_DIR, "posture_model.tflite"), task="pose")
emotion_model = YOLO(os.path.join(MODEL_DIR, "emotion_model.tflite"), task="classify")
sids_model = YOLO(os.path.join(MODEL_DIR, "sids_model.tflite"))

CRIB_LABELS = {0: "baby", 1: "baby_prone", 2: "baby_sideways", 3: "baby_supine"}
SIDS_LABELS = {0: "normal", 1: "covered_face", 2: "overturn"}

EMOTION_LABELS = {
    0: "Angry", 1: "Angry Cry", 2: "Angry Disgust", 3: "Angry Sad", 4: "Cry",
    5: "Cry Disgust", 6: "Cry Neutral", 7: "Cry Neutral Sad", 8: "Disgust",
    9: "Disgust Neutral", 10: "Disgust Sad", 11: "Neutral", 12: "Neutral Sad",
    13: "Sad", 14: "Sad Smile", 15: "Smile", 16: "Smile Surprise",
    17: "Surprise", 18: "Unlabeled",
}

KEYPOINTS = {
    0: "nose", 5: "left_shoulder", 6: "right_shoulder",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
}

motion_detector = MotionDetector()
cap = None
capture_lock = threading.Lock()

_overlap_start_time = None
OVERLAP_ALERT_SECONDS = 30


def get_capture():
    global cap
    with capture_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(PHONE_CAM_URL)
        return cap


def read_frame():
    """Single safe entry point for reading a frame — always use this, never call capture.read() directly."""
    global cap
    capture = get_capture()
    with capture_lock:
        ret, frame = capture.read()
    if not ret or frame is None:
        cap = None  # force reconnect on next call
    return ret, frame


def safe_emotion_label(index):
    return EMOTION_LABELS.get(index, f"unknown_class_{index}")


def bucket_emotion(raw_label):
    lower = raw_label.lower()
    if "cry" in lower:
        return "crying"
    if "sad" in lower or "angry" in lower or "disgust" in lower:
        return "distressed"
    if "smile" in lower:
        return "happy"
    if "neutral" in lower:
        return "calm"
    if "surprise" in lower:
        return "alert"
    return "unknown"


def check_pose_keypoints(kpts):
    if kpts is None or len(kpts) == 0:
        return {"posture": "unknown", "hand_raised": False, "keypoints": None}

    pts = kpts[0]
    nose_y = pts[0][1]
    l_hip_y, r_hip_y = pts[11][1], pts[12][1]
    hips_y = (l_hip_y + r_hip_y) / 2
    l_wrist_y, r_wrist_y = pts[9][1], pts[10][1]
    l_shoulder_y, r_shoulder_y = pts[5][1], pts[6][1]

    posture = "face_down" if nose_y < hips_y else "face_up_or_side"
    hand_raised = bool((l_wrist_y < l_shoulder_y) or (r_wrist_y < r_shoulder_y))

    return {"posture": posture, "hand_raised": hand_raised, "keypoints": pts}


def run_inference():
    global _overlap_start_time

    ret, frame = read_frame()
    if not ret or frame is None:
        raise RuntimeError("Could not read frame from phone camera")

    motion_detected, motion_score = motion_detector.detect(frame)
    results = {"motion": {"detected": motion_detected, "score": motion_score}}

    crib_result = crib_obs_model.predict(frame, verbose=False)[0]
    baby_present = False
    baby_box = None
    crib_posture_label = None

    for box in crib_result.boxes:
        cls_idx = int(box.cls[0])
        label = CRIB_LABELS.get(cls_idx, f"class_{cls_idx}")
        baby_present = True
        baby_box = box.xyxy[0].tolist()
        crib_posture_label = label
        break

    results["crib_obs"] = {"baby_present": baby_present, "posture_label": crib_posture_label}

    if not baby_present:
        results["posture"] = {"status": "skipped_no_baby"}
        results["emotion"] = {"status": "skipped_no_baby"}
        results["sids"] = {"status": "skipped_no_baby"}
        return results

    posture_result = posture_model.predict(frame, verbose=False)[0]
    kpts = posture_result.keypoints.xy.cpu().numpy() if posture_result.keypoints is not None else None
    pose_data = check_pose_keypoints(kpts)

    results["posture"] = {
        "primary_label": crib_posture_label,       # trust this for face-down determination
        "skeleton_hint": pose_data.get("posture"),  # informational only, not authoritative
        "hand_raised": pose_data.get("hand_raised", False),
    }

    try:
        raw_kpts = pose_data.get("keypoints")
        if raw_kpts is not None:
            nose_x, nose_y = raw_kpts[0]
            crop_size = 150
            x1 = max(0, int(nose_x - crop_size / 2))
            y1 = max(0, int(nose_y - crop_size / 2))
            x2 = min(frame.shape[1], int(nose_x + crop_size / 2))
            y2 = min(frame.shape[0], int(nose_y + crop_size / 2))
            head_crop = frame[y1:y2, x1:x2]

            if head_crop.size > 0:
                emotion_result = emotion_model.predict(head_crop, verbose=False)[0]
                probs = emotion_result.probs
                idx = int(probs.top1)
                raw_label = safe_emotion_label(idx)
                results["emotion"] = {
                    "raw_label": raw_label,
                    "category": bucket_emotion(raw_label),
                    "confidence": round(float(probs.top1conf), 3),
                }
            else:
                results["emotion"] = {"status": "crop_failed"}
        else:
            results["emotion"] = {"status": "no_keypoints"}
    except Exception as e:
        results["emotion"] = {"error": str(e)}

    sids_result = sids_model.predict(frame, verbose=False)[0]
    sids_label = "normal"
    for box in sids_result.boxes:
        cls_idx = int(box.cls[0])
        sids_label = SIDS_LABELS.get(cls_idx, f"class_{cls_idx}")
        break

    is_critical = sids_label == "covered_face"

    if is_critical:
        if _overlap_start_time is None:
            _overlap_start_time = time.time()
        elapsed = time.time() - _overlap_start_time
        results["sids"] = {
            "label": sids_label,
            "overlap_seconds": round(elapsed, 1),
            "high_severity": elapsed >= OVERLAP_ALERT_SECONDS,
        }
    else:
        _overlap_start_time = None
        results["sids"] = {"label": sids_label, "high_severity": False}

    return results
