import websockets
import soundfile as sf
import numpy as np
import ast
import asyncio


WS_SERVER_URL = "wss://vad-74416368455.us-central1.run.app/ws"

audio_path = "test_audio.wav"  # Path to your test audio file
chunSize=8  # chunSize/32 s intervals of samples to send to websocket endpoint
audio, sample_rate = sf.read(audio_path, dtype="float32")

async def stream_audio_to_server():
    async with websockets.connect(WS_SERVER_URL) as websocket:
        
        chunk_size = 512*chunSize
        print(f"sending audio to websocket at chunk size of:{chunk_size},which are chunks of intervals of {chunk_size/16000} sec")
        for i in range(0, len(audio), chunk_size):
            # Get a chunk of audio data
            chunk = audio[i:i + chunk_size]
            
            # Convert the chunk to bytes
            audio_bytes = chunk.tobytes()

            # Send the audio chunk to the server
            await websocket.send(audio_bytes)
            
        while True:
            vad_result = await websocket.recv()
            vad_result = ast.literal_eval(vad_result)
            
            print(f"VAD Result in chunk of 512 samples: {vad_result['res']}")  # Print 'speech' or 'no_speech'
            
            if len(vad_result['res'])!=chunSize:
                break

asyncio.run(stream_audio_to_server())

