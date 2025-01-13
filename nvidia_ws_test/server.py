import asyncio
import torch
import websockets
import json
from contextlib import asynccontextmanager
import numpy as np
import time
from silero_vad import (load_silero_vad,
                        read_audio,
                        get_speech_timestamps,
                        save_audio,
                        VADIterator,
                        collect_chunks)
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

# from silero_init import load_silero_vad

vad_model = load_silero_vad(onnx=True)  # Load your VAD model
chunk_size = 512
batch_queue = asyncio.Queue()
batch_size = 32  # Max batch size
max_delay = 0.2  # Max delay in seconds

async def batch_processor():
    while True:
        batch:list[tuple[WebSocket, torch.Tensor]] = []
        start_time = asyncio.get_event_loop().time()  # Start time for the batch

        while True:
            # Check if the max_delay has been exceeded or the batch size is reached
            current_time = asyncio.get_event_loop().time()
            time_elapsed = current_time - start_time

            # print(f'time_elapsed {time_elapsed}')
            # print(f'batch size {getBatchSize(batch)}')
            if time_elapsed >= max_delay or getBatchSize(batch) >= batch_size:
                break  # Process the batch if either condition is met

            try:
                # Wait for a short time to add new items to the batch
                websocket, audio_chunk = await asyncio.wait_for(batch_queue.get(), timeout=max_delay - time_elapsed)
                batch.append((websocket, audio_chunk))
            except asyncio.TimeoutError:
                break

        if batch:  # Only process if there are items in the batch
            try:
                all_chuncks=torch.cat([chunk for _,chunk in batch])
                prob_results = vad_model(all_chuncks, 16000)
                
                responses = []
                # Send results back to clients
                for i in range(len(batch)):
                    result=[ prob.item() for prob in prob_results[0:batch[i][1].size(0)] ]
                    prob_results=prob_results[batch[i][1].size(0):]
                    # if not batch[i][0].application_state == WebSocketState.CONNECTED:
                    responses.append(
                        batch[i][0].send_json({
                            'info':'', 'res':result
                            }))
                await asyncio.gather(*responses)
            except Exception as e:
                print(f"Error: {e}")
                continue

def getBatchSize(batchInfo: list[tuple[WebSocket, torch.Tensor]]):
    return sum(requests.size(0) for _, requests in batchInfo)

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(batch_processor())
    yield


app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunks = torch.tensor(np.frombuffer(data, dtype=np.float32))
            
            num_chunks = audio_chunks.size(0) // chunk_size
            audio_chunks = audio_chunks[:num_chunks * chunk_size].view(num_chunks, chunk_size)#.to('cuda')

            await batch_queue.put((websocket, audio_chunks))
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        vad_model.reset_states()  # Reset states after each audio stream












@app.websocket("/wss")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # vad_iterator = VADIterator(vad_model, sampling_rate=16000)
    try:
        while True:
            # Receive audio chunk from ESP32
            data = await websocket.receive_bytes()
            audio_chunk = torch.tensor(np.frombuffer(data, dtype=np.float32))
            num_chunks = audio_chunk.size(0) // chunk_size
            audio_chunk = audio_chunk[:num_chunks * chunk_size].view(num_chunks, chunk_size)#.to(‘cuda’)
            # VAD processing on the audio chunk
            nw=time.time()
            speech_dict = vad_model(audio_chunk, 16000)
            # [vad_iterator(chunk, return_seconds=True) for chunk in audio_chunk]
            print('test time')
            print(time.time()-nw)
            print(speech_dict)
            # speech_dict=[x for x in speech_dict if x != None]
            # if(len(speech_dict)>0):
            await websocket.send_json(speech_dict.tolist())  # Send result back
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        vad_model.reset_states()  # Reset states after each audio stream

