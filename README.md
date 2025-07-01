# Gunshot Detection with CNN using UrbanSound Dataset

## Project Overview
This project implements a Convolutional Neural Network (CNN) to detect gunshots in audio recordings using the UrbanSound dataset. The model is trained to perform binary classification: distinguishing between "gunshot" and "not gunshot" audio events. The project supports both offline evaluation on audio files and real-time detection from a microphone stream.

## Features
- Binary classification: gunshot vs. not gunshot
- Uses MFCC and Mel-spectrogram features
- 2D CNN architecture for robust audio event detection
- Handles multiple audio formats (wav, mp3, aif, flac)
- Real-time inference from microphone
- Evaluation with confusion matrix and classification report

## Requirements
- Python 3.7+
- [TensorFlow](https://www.tensorflow.org/) (tested with 2.x)
- [NumPy](https://numpy.org/)
- [Librosa](https://librosa.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pydub](https://github.com/jiaaro/pydub)
- [sounddevice](https://python-sounddevice.readthedocs.io/)
- [tqdm](https://tqdm.github.io/)
- [seaborn](https://seaborn.pydata.org/)

Install requirements with:
```sh
pip install tensorflow numpy librosa matplotlib scikit-learn pydub sounddevice tqdm seaborn
```

## Dataset
- [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
- Place the extracted dataset in `./UrbanSound/data/` so that each class is a subfolder (e.g., `./UrbanSound/data/gun_shot/`).

## Usage

### 1. Training
Train the CNN model on the UrbanSound dataset:
```python
# In your notebook or script
# (see main notebook/script for full code)
python train_gunshot_cnn.py
```
The model will be saved as `gunshot_cnn_model.h5` after training.

### 2. Evaluation
Evaluate the model on a test set and visualize results:
- Confusion matrix with TP, FP, TN, FN annotations
- Classification report (precision, recall, F1-score)

### 3. Inference on Audio File
Run inference on a single audio file:
```python
python infer_on_file.py --audio path/to/audio.wav --model gunshot_cnn_model.h5
```

### 4. Real-Time Detection
Detect gunshots from a live microphone stream:
```python
python realtime_gunshot_detection.py --model gunshot_cnn_model.h5
```
The script will print a message when a gunshot is detected.

## Project Structure
- `train_gunshot_cnn.py` — Training and evaluation code
- `realtime_gunshot_detection.py` — Real-time detection from microphone
- `infer_on_file.py` — Inference on a single audio file
- `README.md` — Project documentation
- `UrbanSound/data/` — Dataset directory

## Credits
- UrbanSound8K dataset by Justin Salamon, Christopher Jacoby, and Juan Pablo Bello
- Model and code by [Your Name]

## Project Status

- The Jupyter notebook (`gunshot_training.ipynb`) works for training and testing audio files.
- The Python scripts for real-time inference (`script_realtime_inference.py`, etc.) are **not working for now** (work in progress).

---
Feel free to contribute or open issues for improvements! 