import nemo.collections.asr as nemo_asr

# import asyncio
from typing import AsyncIterator, List
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput, Parameters
from mlserver.codecs import StringCodec, Base64Codec, decode_args
import numpy as np
import soundfile as sf
from io import BytesIO
import torch
import resampy
import base64


class MyKulModel(MLModel):

    async def load(self) -> bool:
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            # "nvidia/parakeet-ctc-1.1b"
            "nvidia/stt_en_fastconformer_ctc_large"
            # "nvidia/stt_en_citrinet_1024_gamma_0_25"
            )
        return True
    
    
    async def predict(
        self, payload: InferenceRequest
    ) -> InferenceResponse:
        # print(f'payload: {payload}')
        # print(f'payload: {payload.model_dump_json()}')
        # print(f'payload: {type(payload.inputs[0].data)}')
        # print(f'payload: {type(payload.inputs[0].data[0])}')

        results=transcribe(payload.inputs[0].data,self.asr_model)

        return InferenceResponse(
                    model_name=self._settings.name,
                    outputs=[
                        ResponseOutput(
                            name="output",
                            datatype="BYTES",
                            shape=[len(results)],
                            data=results,
                            parameters=Parameters(content_type="str",),
                        ),
                    ],
                )



def transcribe(bytesList:List[List[float]] #List[bytes]   currently it's base64 encoded string, use bytes if neccessary
               ,asr_model)->List[str]:
    audios=bytesList
    audioLengths=[len(b) for b in bytesList]
    # for bytes in bytesList:
    #     bytes=base64.b64decode(bytes) # remove if already passing in bytes

    #     audio_data, sample_rate = sf.read(BytesIO(bytes), dtype='float32')
        
    #     target_sample_rate = 16000
    #     if sample_rate != target_sample_rate:
    #         audio_data = resampy.resample(audio_data, sample_rate, target_sample_rate)
    
    #     # Convert stereo to mono if necessary
    #     if len(audio_data.shape) > 1:
    #         audio_data = np.mean(audio_data, axis=1)

    #     print(audio_data.shape)
    #     audios.append(audio_data)
    #     audioLengths.append(audio_data.shape[-1])

        
    audio_data = torch.as_tensor(audios, dtype=torch.float32).to(asr_model.device)

    # Prepare the signal length (batch size is 1, so it's a single value)
    input_signal_length = torch.tensor(audioLengths, dtype=torch.int64).to(asr_model.device)

    # print(audio_data)
    # print(audio_data.shape)
    # print(input_signal_length)
    # The ASR model expects a list of audio inputs
    with torch.no_grad():
        log_probs, encoded_len, predictions = asr_model(input_signal=audio_data, input_signal_length=input_signal_length)#.transcribe(audio_data)

    # print(predictions)
    blank_id=len(asr_model.decoder.vocabulary)

    transcriptions=[]
    for prediction in predictions.cpu().numpy():
        tokens=[tok for tok in prediction if tok != blank_id]

        transcription = asr_model.tokenizer.ids_to_text(list(map(int, tokens)))
        transcriptions.append(transcription)
        # print(transcriptions)

    return transcriptions
