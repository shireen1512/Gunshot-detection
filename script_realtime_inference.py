# pip install sounddevice numpy librosa tensorflow pydub

import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import queue
import threading
import time

# --- CONFIGURATION ---
MODEL_PATH = "gunshot_cnn_model.keras"
SAMPLE_RATE = 22050
DURATION = 2.0  # seconds
MFCC_NUM = 40
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
THRESHOLD = 0.5  # Probability threshold for gunshot detection

# --- LOAD MODEL ---
model = load_model(MODEL_PATH)

# --- FEATURE EXTRACTION (same as training) ---
def extract_features(audio, sr=SAMPLE_RATE, n_mfcc=MFCC_NUM, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = librosa.util.normalize(mfccs)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = librosa.util.normalize(mel)
    features = np.stack([mfccs, mel[:n_mfcc, :]], axis=-1)
    return features

# --- AUDIO STREAMING SETUP ---
q_audio = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    print("indata shape:", indata.shape)
    audio = indata[:, 0]  # or try both channels
    print("audio callback triggered", np.mean(audio))
    q_audio.put(audio.copy())

def listen_and_detect():
    buffer = np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)
    stride = int(SAMPLE_RATE * DURATION // 2)  # 50% overlap for responsiveness

    try:
        with sd.InputStream(device=1, channels=1, samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=stride):
            print("Listening for gunshots (press Ctrl+C to stop)...")
            while True:
                audio_chunk = q_audio.get()
                buffer = np.roll(buffer, -len(audio_chunk))
                buffer[-len(audio_chunk):] = audio_chunk

                features = extract_features(buffer, SAMPLE_RATE)
                features = np.expand_dims(features, axis=0)  # Add batch dimension

                prob = model.predict(features)[0][0]
                if prob > THRESHOLD:
                    print(f"Gunshot detected! (confidence: {prob:.2f})")
                else:
                    print(f"Not gunshot (confidence: {1-prob:.2f})")
    except Exception as e:
        print("Error opening InputStream:", e)

if __name__ == "__main__":
    try:
        listen_and_detect()
    except KeyboardInterrupt:
        print("Stopped by user.")