import cv2
import numpy as np

class MotionDetector:
    def __init__(self, threshold=25, min_area=500):
        self.prev_frame = None
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, frame):
        """Returns (motion_detected: bool, motion_score: float)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.prev_frame = gray

        motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > self.min_area)
        frame_area = frame.shape[0] * frame.shape[1]
        motion_score = min(motion_area / frame_area, 1.0)

        return motion_area > 0, round(motion_score, 3)
