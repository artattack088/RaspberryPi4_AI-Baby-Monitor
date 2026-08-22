# Momora — AI Baby Monitor

Momora is an AI-assisted baby monitoring system for Raspberry Pi. It combines
computer vision (posture, face-covering / SIDS-risk detection, emotion
recognition), audio classification (cry/distress detection), and environmental
sensor readings (temperature, humidity, pressure) into a single fused risk
score, exposed through a FastAPI backend with user authentication and alert
history.

> ⚠️ **Disclaimer:** Momora is an experimental / educational project (built as
> a final year project). It is **not** a certified medical device and must
> **never** be used as a substitute for recommended infant safe-sleep
> practices (e.g. back-to-sleep, firm flat surface, no loose bedding). Always
> follow guidance from a pediatrician or your national health authority.

---

## Features

- 📷 **Vision pipeline** — baby presence & posture detection (crib
  observation model), face-down/prone detection, face-covering detection with
  sustained-overlap timing, facial emotion classification, and basic motion
  detection.
- 🔊 **Audio pipeline** — records ambient audio, filters out silence via an
  energy threshold, and classifies sound into categories (hungry, pain,
  burping, discomfort, cold, tired, noise) using a trained audio model.
- 🌡️ **Environmental sensing** — temperature / humidity / pressure via a
  BME280 sensor, with configurable high/low temperature thresholds.
- 🧠 **Decision engine** — fuses all signals into a single weighted risk
  score (0–1), maps it to a severity level (`low` / `mid` / `high`), and uses
  debouncing (N consecutive confirmations) to avoid single-frame/single-read
  false positives.
- 🔐 **Auth & alerts API** — FastAPI backend with signup/login (JWT), protected
  alert history endpoints, snapshot/recording capture, and a live MJPEG
  stream.
- 🗄️ **Persistence** — SQLite database (via SQLAlchemy) for users, alerts,
  sensor logs, and inference logs.

---

## System Architecture

```
 ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
 │  Camera feed │   │  USB Mic     │   │  BME280 (I2C) │
 └──────┬──────┘   └──────┬──────┘   └──────┬───────┘
        │                 │                  │
        ▼                 ▼                  ▼
 ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
 │ vision_model │   │ audio_model  │   │   sensor.py   │
 │  (YOLO/.tflite)│   │ (YOLO/.tflite)│   │ (bme280 lib) │
 └──────┬──────┘   └──────┬──────┘   └──────┬───────┘
        │                 │                  │
        └────────┬────────┴─────────┬────────┘
                  ▼                  ▼
           decision_engine.py (risk fusion)
                  │
                  ▼
             master.py (background threads + shared state)
                  │
                  ▼
              api.py (FastAPI — auth, alerts, capture, live stream)
```

Background threads (started from `master.py`) continuously poll the sensor,
run vision inference, run audio inference, and re-evaluate the fused risk
score, writing results into a shared in-memory `state` dict that the API
serves.

---

## Hardware Requirements

| Component | Notes |
|---|---|
| Raspberry Pi (3B+ / 4 / 5 recommended) | Runs the full backend + AI inference |
| Camera | See **Camera Setup** below — any of: Pi Camera Module (CSI ribbon), IP/RTSP camera, or phone-as-webcam app |
| USB microphone | Any class-compliant USB mic; auto-detected by name (`"USB Audio Device"`) |
| BME280 sensor | Connected via I2C (default address `0x77`) for temperature/humidity/pressure |
| MicroSD card | 32GB+ recommended (models, recordings, snapshots, DB all stored locally) |

### Camera setup

This project was originally built with a **Raspberry Pi Camera Module**
(CSI ribbon cable, fed through `rpicam-vid` in `config.yml`). In practice the
ribbon connection was unreliable and prone to buffering, so a **phone camera
(via an IP-webcam style app) was used as the working backup** and is what
`PHONE_CAM_URL` in `vision_model.py` points to by default.

Any of the following will work — pick whichever is most reliable on your
setup:
- **Pi Camera Module** — via `rpicam-vid` / `go2rtc` / Frigate (see `config.yml`)
- **Phone as webcam** — any app that exposes an MJPEG/HTTP video stream (e.g.
  IP Webcam on Android, EpocCam/DroidCam) — set its stream URL as
  `PHONE_CAM_URL` in `models/vision_model.py`
- **Any RTSP/IP camera** — set its stream in `config.yml` under `cameras.*.ffmpeg.inputs.path`

If your video feed keeps cutting out, a wired CSI ribbon connection is often
the first suspect — try reseating it or fall back to the phone-cam route
above.

---

## Software Prerequisites

- **OS:** Raspberry Pi OS (Bullseye/Bookworm) or another Debian-based Linux.
  I2C must be enabled (`raspi-config` → Interface Options → I2C).
- **Python:** 3.9+ (3.10/3.11 recommended for `ultralytics`/`tflite-runtime`
  compatibility)
- **System packages** (install before Python deps):
  ```bash
  sudo apt update
  sudo apt install -y python3-pip python3-venv i2c-tools libatlas-base-dev \
      libopenjp2-7 libportaudio2 ffmpeg
  ```
- **External services used for camera ingestion:**
  - [Frigate](https://frigate.video/) — `config.yml` is a Frigate config
    (`detectors`, `cameras`, `motion`, `record`, etc.). Frigate was used
    during development to handle camera ingestion, motion zones, and
    recording/retention (7-day retain, see `config.yml`) for both the Pi
    camera and the phone-cam backup stream.
  - [go2rtc](https://github.com/AlexxIT/go2rtc) — used by Frigate under the
    hood (and directly referenced in `config.yml`) to pull the Pi camera
    stream via `rpicam-vid` and re-serve it.
  - Install/run Frigate separately (Docker is the standard route — see the
    [Frigate install docs](https://docs.frigate.video/frigate/installation/)),
    point `config.yml` at it, and Momora's `vision_model.py` will pull frames
    from the resulting stream URL.

- **Setup automation:** initial setup (system packages, venv, I2C enable,
  Frigate/go2rtc install, service files) was scripted during development for
  repeatable deployment to a fresh Pi, but that automation script is
  environment-specific and isn't included in this repo. Once you've confirmed
  the manual steps above work for your hardware, it's straightforward to wrap
  them into your own `setup.sh` / Ansible playbook / systemd service for
  repeat deployments.

---

## Python Dependencies

See [`requirements.txt`](./requirements.txt). Install with:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note on `tflite-runtime`:** this package isn't always available for the
> newest Python versions via PyPI on all platforms. If installation fails on
> your Pi, check the [TensorFlow Lite runtime install guide](https://www.tensorflow.org/lite/guide/python)
> for a wheel matching your Python version and architecture (`aarch64` for
> 64-bit Pi OS).

---

## Environment Variables

Create a `.env` file in the **project root** — same folder as `api.py`,
`requirements.txt`, etc. (already covered by `.gitignore`, so it won't be
committed):

```env
MOMORA_SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
MOMORA_ALLOWED_ORIGINS=https://your-dashboard-domain.com
```

- `MOMORA_SECRET_KEY` — read by `security.py` at startup; the app should
  refuse to start if it's unset. **Do not** hardcode a secret key in source.
- `MOMORA_ALLOWED_ORIGINS` — read by `api.py` to set CORS `allow_origins`
  instead of the wildcard `"*"`. For local dev, `http://localhost:3000` (or
  whatever your dashboard runs on) is fine; for multiple origins, comma-separate
  them (e.g. `https://dash.example.com,http://192.168.1.50:3000`).

---

## Project Structure

```
momora/
├── api.py                 # FastAPI app: auth, alerts, capture, live stream
├── master.py               # Background thread orchestration + shared state
├── decision_engine.py       # Risk fusion logic
├── database.py              # SQLAlchemy models (User, Alert, SensorLog, InferenceLog)
├── security.py               # Password hashing + JWT handling
├── sensor.py                  # BME280 temperature/humidity/pressure reads
├── media_capture.py            # Snapshot / video clip capture
├── motion_detector.py           # Frame-diff motion detection
├── models/
│   ├── vision_model.py           # Posture, SIDS-risk, emotion inference
│   ├── audio_model.py             # Audio classification
│   ├── crib_obs_model.tflite
│   ├── posture_model.tflite
│   ├── emotion_model.tflite
│   ├── sids_model.tflite
│   └── audio_model.tflite
├── debug_audio_capture.py         # Manual mic/model debug utility
├── config.yml                      # Camera / stream configuration
├── requirements.txt
└── .env                              # (not committed) secrets
```

---

## Running

```bash
source venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000
```

On startup this initializes the database and starts the sensor, vision,
audio, and decision-engine background loops.

### Key API endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/signup` | – | Create a user account |
| POST | `/login` | – | Get a JWT access token |
| GET | `/status` | – | Full current system state |
| GET | `/sensor` / `/vision` / `/audio` / `/risk` | – | Individual state slices |
| GET | `/alerts` | 🔒 | Last 50 alerts |
| POST | `/alerts/{id}/acknowledge` | 🔒 | Mark an alert acknowledged |
| POST | `/capture/snapshot` | 🔒 | Capture a still image |
| POST | `/capture/recording?duration=10` | 🔒 | Record a short clip (max 30s) |
| GET | `/live/stream` | – | MJPEG live stream |

---

## Models

The `.tflite` models in `models/` are YOLO-exported classifiers/detectors
trained for this project (crib/posture observation, SIDS-risk face-covering
detection, emotion classification, and audio-spectrogram classification).
Model training code/data is not included in this repo — only the exported
inference artifacts.

---

## Security Notes

- Rotate `MOMORA_SECRET_KEY` if this repo, or any deployment of it, ever had
  a real key committed to source control.
- CORS `allow_origins` is driven by `MOMORA_ALLOWED_ORIGINS` (see
  **Environment Variables** above) rather than hardcoded — set it to your
  real frontend origin(s) before deploying anywhere reachable from the
  internet. Don't leave it as a wildcard `"*"` outside local dev.
- The SQLite database file (`momora.db`) and anything under `captures/`
  contains real household data (photos/clips/sensor logs) once run — make
  sure these are `.gitignore`d and never committed.

---

## License

MIT License
