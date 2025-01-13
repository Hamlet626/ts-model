from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import spacy
from tortoise.api_fast import TextToSpeech
import io
import wave

# Global variable for the Tortoise model
tts = None
nlp = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the Tortoise model...")
    global tts
    if tts is None:
        print("Loading the Tortoise model...")
        tts = TextToSpeech(use_deepspeed=True, kv_cache=True, half=True)  # Load Tortoise model during startup
    global nlp
    if nlp is None:
        nlp=spacy.load("en_core_web_sm")
    yield

app = FastAPI(lifespan=lifespan)


@app.get('/ping')
def health_check():
    return {"status": "healthy"}

@app.post("/invocations")
async def synthesize_stream(request: Request):
    # return 'test'
    global tts
    req_data = await request.json()
    text = req_data.get("text", "")
    try:
        text_chunks = split_text(text, max_length=200)
        print(text_chunks)

        def stream_audio():
            wav_header = generate_wav_header()
            yield wav_header

            for chunk in text_chunks:
                audio_stream = generate_audio_stream(chunk, tts)
                for audio_chunk in audio_stream:
                    # Convert the audio tensor to bytes (int16 format for WAV)
                    audio_data = audio_chunk.cpu().numpy().astype('int16').tobytes()
                    yield audio_data

        return StreamingResponse(stream_audio(), media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_wav_header(num_channels=1, sample_width=2, frame_rate=22050, n_frames=0):
            """
            Generate WAV file header for the audio stream.
            """
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(sample_width)  # 2 bytes for int16
                wav_file.setframerate(frame_rate)
                wav_file.writeframes(b'')  # Placeholder for the initial header
            return buffer.getvalue()

def generate_audio_stream(text, tts):
    stream = tts.tts_stream(
        text,
        verbose=True,
        stream_chunk_size=40  # Adjust chunk size as needed
    )
    for audio_chunk in stream:
        yield audio_chunk


def split_text(text, max_length=200):
    global nlp
    doc = nlp(text)
    chunks = []
    chunk = []
    length = 0

    for sent in doc.sents:
        sent_length = len(sent.text)
        if length + sent_length > max_length:
            chunks.append(' '.join(chunk))
            chunk = []
            length = 0
        chunk.append(sent.text)
        length += sent_length + 1

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks



# import os
# from fastapi import FastAPI
# import uvicorn

# app = FastAPI()

# @app.get("/ping")
# async def ping():
#     return {"status": "healthy"}

# @app.post("/invocations")
# async def infer():
#     return {"result": "inference result"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))
