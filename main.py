import io
import numpy as np
import torch
torch.set_num_threads(1)
import torchaudio
torchaudio.set_audio_backend("soundfile")
import pyaudio
import threading
import scipy.io.wavfile as wavfile
from gradio_client import Client, file
import time
from silero.utils_vad import init_jit_model
import wave

client = Client("the-cramer-project/Wave2Vec_Kyrgyz")

model = init_jit_model(model_path=f"./silero/files/silero_vad.jit")

# Taken from utils_vad.py
def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound


FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)

audio = pyaudio.PyAudio()

num_samples = 1536

def stop():
    input("Press Enter to stop the recording:")
    global continue_recording
    continue_recording = False

def start_recording():

    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)
    
    global continue_recording
    continue_recording = True
        
    stop_listener = threading.Thread(target=stop)
    stop_listener.start()

    is_probably_akylai = []

    user_recording = []

    with wave.open("./streams/telegram_audio.wav", 'rb') as wf:
        # read data (based on the chunk size)
        data = wf.readframes(1024)

        # play stream (looping from beginning of file to the end)
        while data:
            # writing to the stream is what *actually* plays the sound.
            stream.write(data)
            data = wf.readframes(1024)


        stream.stop_stream()

    stream.start_stream()

    record_from_user = False

    while continue_recording:
    
        audio_chunk = stream.read(num_samples)
    
        # in case you want to save the audio later
        
        audio_int16 = np.frombuffer(audio_chunk, np.int16)

        audio_float32 = int2float(audio_int16)


        # get the confidences and add them to the list to plot them later
        new_confidence = model(torch.from_numpy(audio_float32), 16000).item()
        print(new_confidence)

        if (record_from_user):
            if (new_confidence > 0.5):
                user_recording.append(audio_int16)
                continue

            if (new_confidence < 0.5 and len(user_recording) > 0):
                stream.stop_stream()
                audio_array = np.concatenate(user_recording).astype(np.int16)

                wavfile.write('./streams/recording.wav', SAMPLE_RATE, audio_array)
                # Call to our stt-tts server
                time.sleep(3)
                record_from_user = False
                stream.start_stream()
        else:
            if (new_confidence > 0.8 and not record_from_user):
                is_probably_akylai.append(audio_int16)
                continue

            if (len(is_probably_akylai) > 0):
                stream.stop_stream()
                audio_array = np.concatenate(is_probably_akylai).astype(np.int16)

                wavfile.write('./streams/tt.wav', SAMPLE_RATE, audio_array)
                is_probably_akylai = []

                result = client.predict(
                    file_=file('./streams/tt.wav'),
                    api_name="/predict"
                )

                print("Stt:", result)
                res = result.lower().replace(" ", "")
                if "лай" in res:
                    # Akylai should say "Ou uguvatam"
                    print("Оу")
                    record_from_user = True
                    time.sleep(3)
                
                stream.start_stream()
                    
        


start_recording()

