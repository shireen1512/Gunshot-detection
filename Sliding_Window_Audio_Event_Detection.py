import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import threading
import time

# --- CONFIGURATION ---
MODEL_PATH = "gunshot_cnn_model.keras"
SAMPLE_RATE = 22050
WINDOW_DURATION = 2.0   # seconds (length of audio window for inference)
STRIDE_DURATION = 0.5   # seconds (how often to run inference)
MFCC_NUM = 40
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
THRESHOLD = 0.5         # Probability threshold for gunshot detection

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

# --- AUDIO BUFFER ---
BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
STRIDE_SIZE = int(SAMPLE_RATE * STRIDE_DURATION)
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
buffer_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    if status:
        print(status)
    audio = indata[:, 0]
    with buffer_lock:
        audio_buffer = np.roll(audio_buffer, -len(audio))
        audio_buffer[-len(audio):] = audio

def inference_loop():
    print("Starting inference loop...")
    while True:
        with buffer_lock:
            buffer_copy = np.copy(audio_buffer)
        # Feature extraction
        features = extract_features(buffer_copy, SAMPLE_RATE)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        # Prediction
        prob = model.predict(features, verbose=0)[0][0]
        if prob > THRESHOLD:
            print(f"[{time.strftime('%H:%M:%S')}] Gunshot detected! (confidence: {prob:.2f})")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Not gunshot (confidence: {1-prob:.2f})")
        time.sleep(STRIDE_DURATION)

if __name__ == "__main__":
    print("Listening for gunshots (press Ctrl+C to stop)...")
    try:
        # Start audio stream
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=STRIDE_SIZE):
            # Start inference in a separate thread
            t = threading.Thread(target=inference_loop, daemon=True)
            t.start()
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")