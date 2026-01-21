import pyaudio
import wave
import time
#import keyboard
import torch
import torch.nn as nn
import torchaudio
from torchvision import models
import dataset
import model as pymod
import numpy as np
import matplotlib.pyplot as plt

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
SAMPLE_RATE = 44100
CHUNK_LENGTH_SEK = 1
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_13.pth"

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_LENGTH_SEK)

segment_count = 0
frames = []
start_time = time.time()

#TODO cleanup v
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = models.efficientnet_b0()
#model.classifier[0] = nn.Dropout(p=0.2) #p=0.2 is default
#in_features = model.classifier[1].in_features
#model.classifier[1] = nn.Linear(in_features, 2)

model = pymod.get_model(device)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

while True: 
    data = stream.read(CHUNK, exception_on_overflow=False)
    #frames.append(data)
    audio_np = np.frombuffer(data, dtype=np.float32)
    

    # int16 → float32 in [-1.0, 1.0]
    #audio_np = audio_np.astype(np.float32) / 32768.0
    #print(type(audio_np))
    #print(audio_np.dtype)
    #print("--------------------")
    #print(np.isnan(audio_np).sum())
    #print(np.max(audio_np), np.min(audio_np), np.mean(audio_np), np.std(audio_np))
    #plt.plot(audio_np)
    #plt.show()

    # torch tensor
    audio_tensor = torch.from_numpy(audio_np)

    # torchaudio expects (channels, samples)
    audio_tensor = audio_tensor.unsqueeze(0)
    #print(audio_tensor.shape)

    resample_audio_tensor = dataset.resample(audio_tensor, sample_rate=SAMPLE_RATE, target_sample_rate=10_000)

    

    mel = dataset.convert_waveform_to_spectogram(SAMPLE_RATE, resample_audio_tensor)
    mel[mel.isinf()] = 0
    #print(mel.isnan().sum())
    #print(mel.isinf().sum())
    #plt.plot(mel[1])
    #plt.show()

    #print(mel.shape)
    mel = mel.unsqueeze(0)
    #print(torch.isnan(mel).sum(), torch.isinf(mel).sum())
    #print(mel)
    
    with torch.no_grad():
            output = model(mel).squeeze()
            print(output)
            if output[0] >= output[1]:
                print(output+ "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    """
    if time.time() - start_time > CHUNK_LENGTH_SEK:
        filename = "currenchunk.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        fnames = wf.writeframes()
        print(wf)
        #mel = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_mels=64)(wf).log2().repeat(3, 1, 1)
        wf = np.array([CHANNELS, data])
        wavetensor = torch.from_numpy(wf)
        mel = dataset.convert_waveform_to_spectogram(SAMPLE_RATE, wavetensor)
        with torch.no_grad():
            output = model(mel)
            print(output)
            #TODO Output verarbeiten
    """

stream.stop_stream()
stream.close()
p.terminate()

