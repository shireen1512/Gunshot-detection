import sounddevice as sd
 
print(sd.query_devices())

def callback(indata, frames, time, status):
    print("callback triggered", indata.shape)
with sd.InputStream(device=1, channels=1, samplerate=22050, callback=callback):
    sd.sleep(5000)