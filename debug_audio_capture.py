import sys
sys.path.insert(0, '.')
from models.audio_model import record_audio, audio_to_spectrogram_image, has_sufficient_energy
from PIL import Image

print("Recording 3 seconds... play the crying sound NOW")
audio = record_audio()
sufficient, rms = has_sufficient_energy(audio)
print(f"RMS: {rms}, sufficient energy: {sufficient}")

img_array = audio_to_spectrogram_image(audio)
img = Image.fromarray(img_array)
img.save("debug_spectrogram.png")
print("Saved debug_spectrogram.png — pull this off the Pi and compare visually to a training sample")
