import whisper
import openai
import os
import pyaudio
import wave
from TTS.api import TTS
from playsound import playsound


model = whisper.load_model("tiny")


# from dotenv import load_dotenv

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define prompt and parameters
prompt = "Hello, how can I help you today?"
temperature = 0.7
max_tokens = 50

# Send request to OpenAI API
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    temperature=temperature,
    max_tokens=max_tokens,
)

# Print response from ChatGPT
print(response.choices[0].text.strip())


# record 5 second audio and save to file

# Set parameters for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

# Initialize audio object
audio = pyaudio.PyAudio()

# Open audio stream for recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# Record audio
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# Close audio stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save audio to file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Running a multi-speaker and multi-lingual model

# List available 🐸TTS models and choose the first one
tts = TTS(model_name="tts_models/en/blizzard2013/capacitron-t2-c150_v2", progress_bar=False, gpu=False)
# Run TTS
tts.tts_to_file(text="Hello world.", file_path='test.wav')


# Play audio file
playsound('test.wav')