import asyncio
import json
import time
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Request, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import nemo.collections.asr as nemo_asr
import numpy as np
from openai import OpenAI
import soundfile as sf
from io import BytesIO
import torch
from nemo.collections.asr.models import EncDecClassificationModel
import resampy

from config import OPENAI_API_KEY,system_prompt,system_prompt_json,response_schema

asr_model = None
vad_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the Tortoise model...")
    global asr_model
    if asr_model is None:
        print("Loading the Tortoise model...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            # "nvidia/parakeet-ctc-1.1b"
            "nvidia/stt_en_fastconformer_ctc_large"
            # "nvidia/stt_en_citrinet_1024_gamma_0_25"
            )
        #nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_large")
    global vad_model
    if vad_model is None:
        vad_model=EncDecClassificationModel.from_pretrained(model_name="vad_marblenet")
    yield

app = FastAPI(lifespan=lifespan
              )


@app.get('/ping')
def health_check():
    global asr_model
    return {"status": asr_model is not None}

@app.post("/transcribe")
async def transcribe_audio(request: Request#file: UploadFile = File(...), 
                        #    repeat:int = Form(...)
                           ):
    try:
        # Read the audio data directly from the UploadFile stream (in memory)
        repeat=0
        # file_bytes = await file.read()
        # audio_data, sample_rate = sf.read(BytesIO(file_bytes), dtype='float32')
        
        # target_sample_rate = 16000
        # if sample_rate != target_sample_rate:
        #     audio_data = resampy.resample(audio_data, sample_rate, target_sample_rate)
        
        # # return {"data":audio_data.tolist()}
        # # audio_data=audio_data[:128000]
        # # Convert stereo to mono if necessary
        # if len(audio_data.shape) > 1:
        #     audio_data = np.mean(audio_data, axis=1)
        data = await request.body()
        audio_data = np.frombuffer(data, dtype=np.float32)

        global asr_model
        audio_data = torch.as_tensor(np.repeat([audio_data], repeats=repeat, axis=0) if repeat!=0 else [audio_data], 
                                     dtype=torch.float32).to(asr_model.device)

        # Prepare the signal length (batch size is 1, so it's a single value)
        input_signal_length = torch.tensor(np.repeat([audio_data.shape[1]], repeats=repeat, axis=0) if repeat!=0 else [audio_data.shape[1]], 
                                           dtype=torch.int64).to(asr_model.device)

        # print(audio_data)
        # print(audio_data.shape)
        # print(input_signal_length)
        # The ASR model expects a list of audio inputs
        with torch.no_grad():
            log_probs, encoded_len, predictions = asr_model(input_signal=audio_data, input_signal_length=input_signal_length)#.transcribe(audio_data)

        # print(predictions)
        # print(predictions.cpu().numpy().shape)
        blank_id=len(asr_model.decoder.vocabulary)
        predictions=[tok for tok in predictions.cpu().numpy()[0] if tok != blank_id]

        transcriptions = asr_model.tokenizer.ids_to_text(list(map(int, predictions)))
        # print(transcriptions)
        return {"transcription": transcriptions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from google.cloud import texttospeech, texttospeech_v1
import itertools
from google.oauth2 import service_account
openAIclient = OpenAI(api_key=OPENAI_API_KEY)
# credentials = service_account.Credentials.from_service_account_file("sparko.json")
tts_client = texttospeech.TextToSpeechClient()

@app.websocket("/reply")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    botid = websocket.query_params.get("uid")
    stream_voice = websocket.query_params.get("paid")=='1'

    try:
        user_audio = await websocket.receive_bytes()
        user_text = stt(user_audio)
        for audio_chunk in (qa_stream(user_text)if stream_voice else qa_std(user_text)):
            await websocket.send_bytes(audio_chunk)
        await websocket.send_text('end')
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"error : {e}")


def stt(speech:bytes)->str:
    print(f'in stt{len(speech)}')
    audio_data = np.frombuffer(speech, dtype=np.float32)

    print(f'in stt2{len(audio_data)}')
    global asr_model
    audio_data = torch.as_tensor([audio_data], 
                                    dtype=torch.float32).to(asr_model.device)

    input_signal_length = torch.tensor([audio_data.shape[1]], 
                                        dtype=torch.int64).to(asr_model.device)

    with torch.no_grad():
        log_probs, encoded_len, predictions = asr_model(input_signal=audio_data, input_signal_length=input_signal_length)#.transcribe(audio_data)

    blank_id=len(asr_model.decoder.vocabulary)
    predictions=[tok for tok in predictions.cpu().numpy()[0] if tok != blank_id]

    print(f'in stt 3{predictions}')
    transcriptions = asr_model.tokenizer.ids_to_text(list(map(int, predictions)))
    print(transcriptions)
    return transcriptions

def qa_stream(prompt:str):
    # See https://cloud.google.com/text-to-speech/docs/voices for all voices.
    streaming_config = texttospeech.StreamingSynthesizeConfig(
        # audio_encoding=texttospeech.AudioEncoding.MP3,
        voice=texttospeech.VoiceSelectionParams(name="en-US-Journey-D", language_code="en-US"))

    # Set the config for your stream. The first request must contain your config, and then each subsequent request must contain text.
    config_request = texttospeech.StreamingSynthesizeRequest(
        streaming_config=streaming_config)

    def request_generator():
        stream=openAIclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system","content": system_prompt},
                {"role": "user", "content": prompt}],
            stream=True
        )
        for response in stream:
            if response.choices[0].delta.content is not None:
                partial_text = response.choices[0].delta.content
                print(partial_text)
                if(partial_text):
                    yield texttospeech.StreamingSynthesizeRequest(input=texttospeech.StreamingSynthesisInput(text=partial_text))

    streaming_responses = tts_client.streaming_synthesize(
        requests=itertools.chain([config_request], request_generator()))
    for response in streaming_responses:
        yield response.audio_content


def qa_std(prompt:str):
    reply:ChatCompletion=openAIclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system","content": system_prompt_json},
                {"role": "user", "content": prompt}],
            stream=False,
            response_format={'type':'json_schema','json_schema':response_schema}
            # response_format={ "type":"json_object" }
        )
    
    reply_obj=json.loads(reply.choices[0].message.content)

    synthesis_input = texttospeech.SynthesisInput(text=reply_obj['text_reply'])

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name='en-US-Standard-D',
        # ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    yield response.audio_content


@app.post("/vad")
async def vad_endpoint(request: Request):

    data = await request.body()
    audio_data = np.frombuffer(data, dtype=np.float32)
    frameSize=12000 if request.headers['frame'] is None else int(request.headers['frame'])
    thres=0.6 if request.headers['thres'] is None else float(request.headers['thres'])

    # Perform VAD in chunks if necessary
    audio_chunks = np.array_split(audio_data, len(audio_data) // frameSize)  # 0.75-second chunks
    voice_segments = []
    prob_list=[]
    global vad_model
    for chunk in audio_chunks:
        inputs = torch.tensor(chunk).unsqueeze(0).to(vad_model.device)  # Add batch dimension
        with torch.no_grad():
            logits = vad_model.forward(input_signal=inputs, input_signal_length=torch.tensor([len(chunk)]))
            probs = torch.softmax(logits[0], dim=-1)
            prob_list.append(probs[1].item())
            voice_activity = probs[1].item() > thres  # threshold for VAD
            voice_segments.append(voice_activity)

    return {"voice_segments": voice_segments,"probs":prob_list}