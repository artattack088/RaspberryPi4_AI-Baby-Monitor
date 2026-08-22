"""
Momora Decision Engine

Combines outputs from vision models, audio model, and environmental
sensor readings into a single weighted risk score, then maps that
score to an alert severity (low / mid / high).

Temperature thresholds (infant room safety guidance):
  Normal range: 20-22°C
  Low warning:  < 18°C  (risk of hypothermia)
  High warning: > 26°C  (risk of overheating, linked to SIDS risk factors)

Debounce: posture and motion signals must be confirmed over N consecutive
inference cycles before contributing to risk, to avoid single-frame
false positives.

Thresholds (tunable, documented for FYP evaluation):
  risk_score < 0.1          -> no alert (all clear)
  0.1  <= risk_score < 0.4  -> LOW
  0.4  <= risk_score < 0.7  -> MID
  risk_score >= 0.7         -> HIGH
"""

from collections import deque

POSTURE_CONSECUTIVE_REQUIRED = 4
MOTION_CONSECUTIVE_REQUIRED = 2
TEMP_CONSECUTIVE_REQUIRED = 3   # avoid alerting on a single noisy sensor read

TEMP_LOW_THRESHOLD = 18.0
TEMP_HIGH_THRESHOLD = 26.0

_posture_history = deque(maxlen=POSTURE_CONSECUTIVE_REQUIRED)
_motion_history = deque(maxlen=MOTION_CONSECUTIVE_REQUIRED)
_temp_low_history = deque(maxlen=TEMP_CONSECUTIVE_REQUIRED)
_temp_high_history = deque(maxlen=TEMP_CONSECUTIVE_REQUIRED)


def evaluate(vision_results: dict, audio_result: dict, sensor_result: dict = None):
    contributions = {}
    messages = []

    # --- SIDS ---
    sids = vision_results.get("sids", {})
    if sids.get("label") == "covered_face":
        if sids.get("high_severity"):
            contributions["sids"] = 0.40
            messages.append(f"CRITICAL: Face covered for {sids.get('overlap_seconds')}s")
        else:
            contributions["sids"] = 0.18
            messages.append("Face covering detected, monitoring")
    elif sids.get("label") == "overturn":
        contributions["sids"] = 0.15
        messages.append("Baby rolled onto stomach")

    # --- POSTURE: driven by crib_obs classifier (purpose-trained) ---
    posture = vision_results.get("posture", {})
    is_face_down = bool(posture.get("primary_label") == "baby_prone")
    _posture_history.append(is_face_down)
    posture_confirmed = bool(
        len(_posture_history) == POSTURE_CONSECUTIVE_REQUIRED and all(_posture_history)
    )
    if posture_confirmed:
        contributions["posture"] = 0.18
        messages.append("Sustained face-down (prone) posture")

    # --- EMOTION ---
    emotion = vision_results.get("emotion", {})
    category = emotion.get("category")
    if category in ("crying", "distressed") and emotion.get("confidence", 0) >= 0.6:
        contributions["emotion"] = 0.15
        messages.append(f"Emotion: {emotion.get('raw_label')}")

    # --- AUDIO ---
    if audio_result.get("confirmed"):
        audio_label = audio_result.get("label", "distress")
        contributions["audio"] = 0.18
        messages.append(f"Audio: {audio_label}")

    # --- TEMPERATURE (new) ---
    if sensor_result and sensor_result.get("status") == "ok":
        temp = sensor_result.get("data", {}).get("temperature")
        if temp is not None:
            is_too_low = bool(temp < TEMP_LOW_THRESHOLD)
            is_too_high = bool(temp > TEMP_HIGH_THRESHOLD)

            _temp_low_history.append(is_too_low)
            _temp_high_history.append(is_too_high)

            low_confirmed = bool(
                len(_temp_low_history) == TEMP_CONSECUTIVE_REQUIRED and all(_temp_low_history)
            )
            high_confirmed = bool(
                len(_temp_high_history) == TEMP_CONSECUTIVE_REQUIRED and all(_temp_high_history)
            )

            if high_confirmed:
                contributions["temperature"] = 0.20
                messages.append(f"Room too warm: {temp}°C (threshold: {TEMP_HIGH_THRESHOLD}°C)")
            elif low_confirmed:
                contributions["temperature"] = 0.20
                messages.append(f"Room too cold: {temp}°C (threshold: {TEMP_LOW_THRESHOLD}°C)")

    # --- DEMO FALLBACK: generic motion ---
    motion = vision_results.get("motion", {})
    _motion_history.append(bool(motion.get("detected", False)))
    motion_confirmed = bool(
        len(_motion_history) == MOTION_CONSECUTIVE_REQUIRED and all(_motion_history)
    )
    if motion_confirmed and not contributions:
        contributions["motion"] = 0.10
        messages.append("Movement detected")

    risk_score = round(sum(contributions.values()), 3)

    if risk_score >= 0.7:
        severity = "high"
    elif risk_score >= 0.4:
        severity = "mid"
    elif risk_score >= 0.1:
        severity = "low"
    else:
        severity = None

    return {
        "risk_score": risk_score,
        "severity": severity,
        "triggered_by": list(contributions.keys()),
        "message": " | ".join(messages) if messages else "All clear",
    }
