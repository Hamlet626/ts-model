import asyncio
import websockets

# URL for the WebSocket server (replace with your Cloud Run URL)
WS_SERVER_URL = "wss://tts-test-74416368455.us-central1.run.app/reply"
# "ws://0.0.0.0:8080/ws/generate-audio"

async def stream_audio_to_server():
    async with websockets.connect(WS_SERVER_URL) as websocket:
        await websocket.send('what color is the sky')
        vad_result=''
        # the backend will send text 'end' when the audio finishes
        while vad_result!='end':
            vad_result = await websocket.recv()
            print(f"response chunk bytes length: {len(vad_result)}")  # Print 'speech' or 'no_speech'
            

import time
nw=time.time()
asyncio.run(stream_audio_to_server())

print(time.time()-nw)

