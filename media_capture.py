import cv2
import os
import time
from datetime import datetime

MEDIA_DIR = os.path.join(os.path.dirname(__file__), "captures")
SNAPSHOT_DIR = os.path.join(MEDIA_DIR, "snapshots")
RECORDING_DIR = os.path.join(MEDIA_DIR, "recordings")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(RECORDING_DIR, exist_ok=True)


def take_snapshot(capture):
    ret, frame = capture.read()
    if not ret or frame is None:
        raise RuntimeError("Could not read frame for snapshot")

    filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(filepath, frame)

    return {"filename": filename, "path": filepath}


def record_clip(capture, duration_seconds=10):
    filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = os.path.join(RECORDING_DIR, filename)

    ret, frame = capture.read()
    if not ret:
        raise RuntimeError("Could not read frame for recording")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filepath, fourcc, 15.0, (width, height))

    start = time.time()
    frame_count = 0
    while time.time() - start < duration_seconds:
        ret, frame = capture.read()
        if not ret:
            break
        writer.write(frame)
        frame_count += 1
        time.sleep(1 / 15)

    writer.release()

    return {"filename": filename, "path": filepath, "duration": duration_seconds, "frames": frame_count}
