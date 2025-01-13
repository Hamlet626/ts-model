#!/usr/bin/env python3

import asyncio
import websockets
import wave

async def run_test(uri):
    async with websockets.connect(uri) as websocket:

        print('test debug')
        wf = wave.open('speech_test.mp3', "rb")
        await websocket.send('{ "config" : { "sample_rate" : %d } }' % (wf.getframerate()))
        buffer_size = int(wf.getframerate() * 0.2) # 0.2 seconds of audio
        print(buffer_size)
        while True:
            data = wf.readframes(buffer_size)

            if len(data) == 0:
                break

            await websocket.send(data)
            print (await websocket.recv())

        await websocket.send('{"eof" : 1}')
        print (await websocket.recv())

# asyncio.run(run_test('wss://tts-test-74416368455.us-central1.run.app'))
asyncio.run(run_test('ws://35.224.103.247:2700'))