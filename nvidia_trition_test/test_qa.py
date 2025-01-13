import asyncio
import numpy as np
import websockets
import soundfile as sf

# URL for the WebSocket server 
#  'uid' will be the real ESP device UID
#  'paid' is a parameter for faster and streaming response, it's just for testing now, 
#         and will be inferred from uid on backend later in production.
#         '1' for streaming response, other for normal response (but both will be denoted by 'end' for end of the audio messages)
device_uid='ESP32_UID'
WS_SERVER_URL = f"wss://tts-test-74416368455.us-central1.run.app/reply?uid={device_uid}&paid=1"

output_file = "output_audio.wav"   # Path for response audio file
audio_path = "sky_clear_16000.wav"  # Path for user input audio file, in 16000Hz


audio, sample_rate = sf.read(audio_path, dtype="float32")

async def stream_audio_to_server():
    async with websockets.connect(WS_SERVER_URL) as websocket:
        await websocket.send(audio.tobytes())
        response=''

        # response is the encoded audio data bytes.
        # This is headerless LINEAR16 audio with a sample rate of 24000.
        with sf.SoundFile(output_file, mode='w', samplerate=24000, channels=1, subtype='PCM_16') as wav_file:
            # the backend will send text 'end' when the audio finishes
            while response!='end':
                response = await websocket.recv()
                if response != 'end':
                    # Decode LINEAR16 data to int16
                    audio_chunk = np.frombuffer(response, dtype=np.int16)
                    # Write the chunk to the WAV file
                    wav_file.write(audio_chunk)
                print(f"response chunk bytes length: {len(response)}")
            

asyncio.run(stream_audio_to_server())

