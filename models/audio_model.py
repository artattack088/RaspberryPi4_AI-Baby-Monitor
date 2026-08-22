import numpy as np
import os
import io
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image
from collections import deque
import sounddevice as sd
from ultralytics import YOLO

MODEL_DIR = os.path.dirname(__file__)
audio_model = YOLO(os.path.join(MODEL_DIR, "audio_model.tflite"), task="classify")

SAMPLE_RATE = 22050
DURATION = 3.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

ENERGY_THRESHOLD = 0.016
CONSECUTIVE_REQUIRED = 3
_recent_hits = deque(maxlen=CONSECUTIVE_REQUIRED)

AUDIO_LABELS = {
    0: "hungry",
    1: "pain",
    2: "burping",
    3: "discomfort",
    4: "cold",
    5: "tired",
    6: "noise",
}


def get_usb_mic_index():
    """Finds the USB mic by name so it self-corrects if ALSA device indices shift."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if "USB Audio Device" in d['name'] and d['max_input_channels'] > 0:
            return i
    return None  # falls back to system default if not found


AUDIO_DEVICE_INDEX = get_usb_mic_index()
SAMPLE_RATE = 22050       # what the MODEL expects
HARDWARE_SAMPLE_RATE = 44100  # update this after checking your mic's actual default_samplerate above

def record_audio():
    audio = sd.rec(
        int(DURATION * HARDWARE_SAMPLE_RATE),
        samplerate=HARDWARE_SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=AUDIO_DEVICE_INDEX
    )
    sd.wait()
    audio = audio.flatten()

    # Resample down to what the model expects
    if HARDWARE_SAMPLE_RATE != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=HARDWARE_SAMPLE_RATE, target_sr=SAMPLE_RATE)

    return audio


def has_sufficient_energy(audio, threshold=ENERGY_THRESHOLD):
    audio_clipped = np.clip(audio, -1.0, 1.0)  # guard against overflow from mic glitches
    rms = np.sqrt(np.mean(audio_clipped.astype(np.float64) ** 2))  # float64 avoids overflow
    return rms > threshold, round(float(rms), 4)


def audio_to_spectrogram_image(audio):
    """Recreates the matplotlib magma-colormap spectrogram image used in training."""
    S = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)  # renders to ~224x224
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    librosa.display.specshow(S_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, cmap='magma', ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert('RGB').resize((224, 224))
    return np.array(img)


def run_inference():
    audio = record_audio()
    sufficient, rms = has_sufficient_energy(audio)

    if not sufficient:
        _recent_hits.append(False)
        return {"label": "quiet", "confidence": 0.0, "rms": rms, "skipped": "low_energy", "confirmed": False}

    spec_image = audio_to_spectrogram_image(audio)
    results = audio_model.predict(spec_image, verbose=False)[0]

    probs = results.probs
    label_index = int(probs.top1)
    confidence = float(probs.top1conf)
    label_name = AUDIO_LABELS.get(label_index, f"unknown_{label_index}")

    is_hit = confidence >= 0.75 and label_name != "noise"
    _recent_hits.append(is_hit)
    confirmed = len(_recent_hits) == CONSECUTIVE_REQUIRED and all(_recent_hits)

    return {
        "label": label_name,
        "confidence": round(confidence, 3),
        "rms": rms,
        "confirmed": confirmed,
    }
