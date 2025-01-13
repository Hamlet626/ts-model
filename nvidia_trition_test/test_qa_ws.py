import asyncio
import numpy as np
import websockets
import soundfile as sf
import sounddevice as sd
import time

# URL for the WebSocket server (replace with the real ESP device UID)
device_uid='ESP32_UID'
WS_SERVER_URL = f"ws://0.0.0.0:8080/reply?uid={device_uid}&paid=10"
f"wss://tts-test-74416368455.us-central1.run.app/reply?uid={device_uid}"

output_file = "output_audio.wav" 
audio_path = "sky_clear_16000.wav"  # Path to your test audio file
audio, sample_rate = sf.read(audio_path, dtype="float32")


async def stream_audio_to_server():
    async with websockets.connect(WS_SERVER_URL) as websocket:
        await websocket.send(audio.tobytes())
        vad_result=''
        nw=time.time()
        # the backend will send text 'end' when the audio finishes
        # while vad_result=='':
        #     vad_result = await websocket.recv()
        #     print(f"response chunk bytes length: {len(vad_result)}")  # Print 'speech' or 'no_speech'
        
        print(time.time()-nw)
        nw=time.time()
        with sf.SoundFile(output_file, mode='w', samplerate=24000, channels=1, subtype='PCM_16') as wav_file:
            while vad_result!='end':
                # vad_result is the encoded audio data bytes.
                # This is headerless LINEAR16 audio with a sample rate of 24000.
                vad_result = await websocket.recv()
                # audio_chunk = np.frombuffer(vad_result, dtype=np.int16).astype(np.float32) / 32768.0
                # sd.play(audio_chunk, samplerate=24000)
                # sd.wait()
                if vad_result != 'end':
                    print(vad_result)
                    # Decode LINEAR16 data to int16
                    audio_chunk = np.frombuffer(vad_result, dtype=np.int16)
                    # Write the chunk to the WAV file
                    wav_file.write(audio_chunk)
                print(f"response chunk bytes length: {len(vad_result)}")  # Print 'speech' or 'no_speech'
        print(time.time()-nw)
            

asyncio.run(stream_audio_to_server())

