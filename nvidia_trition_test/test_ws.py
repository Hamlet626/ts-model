import asyncio
import websockets
import soundfile as sf
import numpy as np
import ast
# from silero_vad import (load_silero_vad,
#                         read_audio,
#                         get_speech_timestamps,
#                         save_audio,
#                         VADIterator,
#                         collect_chunks)
# wav = read_audio(f'kis_color_sky_short16000.wav', sampling_rate=16000)
# print(wav.type())
# print(wav.dtype)
# print(wav.shape)
# print(wav)

# URL for the WebSocket server (replace with your Cloud Run URL)
WS_SERVER_URL = "wss://vad-74416368455.us-central1.run.app/ws"
#"ws://0.0.0.0:8080/ws"
audio_path = "kis_long16000.wav"  # Path to your test audio file
chunSize=32
audio, sample_rate = sf.read(audio_path, dtype="float32")

async def stream_audio_to_server():
    async with websockets.connect(WS_SERVER_URL) as websocket:
        # Read audio file and split it into small chunks
        
        # Assuming each chunk represents 20 ms of audio data
        chunk_size = 512*chunSize  # 12/32 s worth of samples
        print(chunk_size)
        for i in range(0, len(audio), chunk_size):
            # Get a chunk of audio data
            chunk = audio[i:i + chunk_size]
            
            # Convert the chunk to bytes
            audio_bytes = chunk.tobytes()

            # Send the audio chunk to the server
            await websocket.send(audio_bytes)
            
            # Wait to receive the VAD result
            # vad_result = await websocket.recv()
            # if(vad_result != '[None]'):
            #     print(f"VAD Result: {vad_result}")  # Print 'speech' or 'no_speech'

        while True:
            vad_result = await websocket.recv()
            vad_result = ast.literal_eval(vad_result)
            # print(f"{type(vad_result)}")
            print(f"VAD Result in chunk of 512 samples: {vad_result['res']}")  # Print 'speech' or 'no_speech'
            
            if len(vad_result['res'])!=chunSize:
                break
            # Simulate real-time streaming delay
            # await asyncio.sleep(0.02)  # 20 ms delay

# Run the client
import time
nw=time.time()
asyncio.run(stream_audio_to_server())

print(time.time()-nw)

