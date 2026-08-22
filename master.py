import threading
import time
import traceback
import logging

from database import SessionLocal, init_db, Alert, SensorLog, InferenceLog
import decision_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("momora")

init_db()

state = {
    "sensor": {"status": "starting"},
    "vision": {"status": "starting"},
    "audio": {"status": "starting"},
    "risk": {"status": "starting"},
}


_last_logged_severity = None
_last_logged_time = 0

def log_alert(result):
    global _last_logged_severity, _last_logged_time
    import time as _time

    if not result["severity"]:
        _last_logged_severity = None
        return

    now = _time.time()
    # Only log if severity changed, or 60s passed since last log of same severity
    if result["severity"] == _last_logged_severity and (now - _last_logged_time) < 60:
        return

    db = SessionLocal()
    try:
        alert = Alert(
            source="ai_fusion",
            alert_type="risk_assessment",
            severity=result["severity"],
            confidence=result["risk_score"],
            risk_score=result["risk_score"],
            message=result["message"],
        )
        db.add(alert)
        db.commit()
        log.warning(f"ALERT LOGGED: {result['message']}")
        _last_logged_severity = result["severity"]
        _last_logged_time = now
    finally:
        db.close()


def run_sensor_loop():
    while True:
        try:
            from sensor import read_sensor
            data = read_sensor()
            state["sensor"] = {"status": "ok", "data": data}

            db = SessionLocal()
            try:
                db.add(SensorLog(
                    temperature=data["temperature"],
                    humidity=data["humidity"],
                    pressure=data["pressure"],
                ))
                db.commit()
            finally:
                db.close()

        except Exception as e:
            state["sensor"] = {"status": "error", "error": str(e)}
            log.error(f"Sensor module crashed: {e}\n{traceback.format_exc()}")
        time.sleep(5)


latest_vision = {}
latest_audio = {}


def run_vision_loop():
    global latest_vision
    while True:
        try:
            from models.vision_model import run_inference
            result = run_inference()
            latest_vision = result
            state["vision"] = {"status": "ok", "data": result}
        except Exception as e:
            state["vision"] = {"status": "error", "error": str(e)}
            log.error(f"Vision module crashed: {e}\n{traceback.format_exc()}")
        time.sleep(1)


def run_audio_loop():
    global latest_audio
    while True:
        try:
            from models.audio_model import run_inference
            result = run_inference()
            latest_audio = result
            state["audio"] = {"status": "ok", "data": result}
        except Exception as e:
            state["audio"] = {"status": "error", "error": str(e)}
            log.error(f"Audio module crashed: {e}\n{traceback.format_exc()}")
        time.sleep(2)


def run_decision_loop():
    while True:
        try:
            if latest_vision and latest_audio:
                result = decision_engine.evaluate(latest_vision, latest_audio, state.get("sensor"))
                state["risk"] = {"status": "ok", "data": result}
                log_alert(result)
        except Exception as e:
            state["risk"] = {"status": "error", "error": str(e)}
            log.error(f"Decision engine crashed: {e}\n{traceback.format_exc()}")
        time.sleep(2)


def start_all_threads():
    threads = [
        threading.Thread(target=run_sensor_loop, daemon=True),
        threading.Thread(target=run_vision_loop, daemon=True),
        threading.Thread(target=run_audio_loop, daemon=True),
        threading.Thread(target=run_decision_loop, daemon=True),
    ]
    for t in threads:
        t.start()
    log.info("Momora master threads started: sensor, vision, audio, decision engine")


if __name__ == "__main__":
    start_all_threads()
    while True:
        time.sleep(10)
