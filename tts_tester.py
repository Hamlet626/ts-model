import os
import time
import torch
import torchaudio
import traceback
# import scipy

# print(scipy.__version__)

# from scipy.stats import betabinom
# betabinom()


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.api import TTS
from TTS.utils.generic_utils import get_user_data_dir

app = FastAPI()

print("Loading model...")
current_dir = os.path.dirname(os.path.abspath(__file__))
config = XttsConfig(audio=XttsAudioConfig(),languages=['en'])
# config.load_json(os.path.join(current_dir, "config", "config.json"))
model = Xtts.init_from_config(config)
# checkpointPath=os.path.join(get_user_data_dir("tts"), "tts_models/multilingual/multi-dataset/xtts_v2".replace("/", "--"))
# print(checkpointPath)
model.load_checkpoint(config, checkpoint_dir='model',
                       use_deepspeed=True)
model.cpu()
# model.cuda()

# model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# model.buffers

@app.get('/health')
def health_check():
    return "Healthy"

@app.post("/synthesize-stream")
async def synthesize_stream(text: str):
    try:
        files = os.listdir()
        print("Files and directories in current directory:", files)
        print("Current working directory:", os.getcwd())
        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[os.path.join(current_dir,"reference.wav")])

        print("Starting inference...")
        chunks = model.inference_stream(
            text,
            "en",
            gpt_cond_latent,
            speaker_embedding
        )

        def stream_audio():
            wav_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    print(f"Time to first chunk: {time.time() - time.time()}")
                print(f"Sending chunk {i} of audio length {chunk.shape[-1]}")
                wav_chunks.append(chunk)
                yield chunk.cpu().numpy().tobytes()
            
            wav = torch.cat(wav_chunks, dim=0)
            torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)

        return StreamingResponse(stream_audio(), media_type="audio/wav")

    except Exception as e:
        traceback.print_exc(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))
