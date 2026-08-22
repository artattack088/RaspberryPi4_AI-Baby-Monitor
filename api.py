from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

import master
from database import get_db, User, Alert, init_db
from security import hash_password, verify_password, create_access_token, decode_access_token

from fastapi.responses import FileResponse
from media_capture import take_snapshot, record_clip
from models.vision_model import get_capture
import os

from fastapi.responses import StreamingResponse
import cv2
import time

# from models.vision_model import capture_lock

app = FastAPI(title="Momora Baby Monitor API")

ALLOWED_ORIGINS = os.environ.get("MOMORA_ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


class SignupRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


@app.on_event("startup")
def startup():
    init_db()
    master.start_all_threads()


# ---------- AUTH ----------

@app.post("/signup")
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(
        (User.username == payload.username) | (User.email == payload.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already registered")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    return {"message": "User created successfully"}


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token({"sub": user.username, "user_id": user.id})
    return {"access_token": token, "token_type": "bearer"}


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(User).filter(User.id == payload.get("user_id")).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ---------- STATUS (public — dashboard can poll without login for demo ease) ----------

@app.get("/")
def root():
    return {"system": "Momora Baby Monitor", "status": "online"}


@app.get("/status")
def full_status():
    return master.state


@app.get("/sensor")
def sensor():
    return master.state["sensor"]


@app.get("/vision")
def vision():
    return master.state["vision"]


@app.get("/audio")
def audio():
    return master.state["audio"]


@app.get("/risk")
def risk():
    return master.state["risk"]


# ---------- ALERTS (protected — requires login) ----------

@app.get("/alerts")
def get_alerts(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).limit(50).all()
    return [
        {
            "id": a.id,
            "severity": a.severity,
            "message": a.message,
            "confidence": a.confidence,
            "risk_score": a.risk_score,
            "timestamp": a.timestamp.isoformat(),
            "acknowledged": bool(a.acknowledged),
        }
        for a in alerts
    ]


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.acknowledged = 1
    db.commit()
    return {"message": "Alert acknowledged"}

# ---------- CAPTURE ----------

@app.post("/capture/snapshot")
def capture_snapshot(current_user: User = Depends(get_current_user)):
    try:
        capture = get_capture()
        result = take_snapshot(capture)
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/capture/recording")
def capture_recording(duration: int = 10, current_user: User = Depends(get_current_user)):
    if duration > 30:
        raise HTTPException(status_code=400, detail="Max recording duration is 30 seconds")
    try:
        capture = get_capture()
        result = record_clip(capture, duration_seconds=duration)
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captures/list")
def list_captures(current_user: User = Depends(get_current_user)):
    from media_capture import SNAPSHOT_DIR, RECORDING_DIR
    return {
        "snapshots": sorted(os.listdir(SNAPSHOT_DIR), reverse=True),
        "recordings": sorted(os.listdir(RECORDING_DIR), reverse=True),
    }


@app.get("/captures/snapshot/{filename}")
def get_snapshot(filename: str, current_user: User = Depends(get_current_user)):
    from media_capture import SNAPSHOT_DIR
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)


@app.get("/captures/recording/{filename}")
def get_recording(filename: str, current_user: User = Depends(get_current_user)):
    from media_capture import RECORDING_DIR
    filepath = os.path.join(RECORDING_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)


@app.get("/live/stream")
def live_stream():
    from models.vision_model import read_frame

    boundary = "frame"

    def generate():
        while True:
            ret, frame = read_frame()
            if not ret or frame is None:
                time.sleep(0.5)
                continue

            resized = cv2.resize(frame, (480, 360))
            ret2, jpeg = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret2:
                continue

            frame_bytes = jpeg.tobytes()

            yield (
                f"--{boundary}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(frame_bytes)}\r\n\r\n"
            ).encode() + frame_bytes + b"\r\n"

            time.sleep(1/10)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}"
    )
