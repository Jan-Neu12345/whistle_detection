import pyaudio
import wave
import time
import keyboard
import torch
import torch.nn as nn
import torchaudio
from torchvision import models

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 44100
CHUNK_LENGTH_SEK = 1
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_13.pth"

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_LENGTH_SEK)

segment_count = 0
frames = []
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0()
#model.classifier[0] = nn.Dropout(p=0.2) #p=0.2 is default
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

while True: 
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

    if time.time() - start_time > CHUNK_LENGTH_SEK:
        filename = "currenchunk.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        mel = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_mels=64)(wf).log2().repeat(3, 1, 1)
        with torch.no_grad():
            output = model(mel)
            #TODO Output verarbeiten

    if keyboard.is_pressed("q"):
        print("Beende...")
        break

stream.stop_stream()
stream.close()
p.terminate()

