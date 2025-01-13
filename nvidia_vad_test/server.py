from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Request
# import torchvision
from contextlib import asynccontextmanager
# import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
from io import BytesIO
import torch
# from nemo.core.classes import IterableDataset
# from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader
from nemo.collections.asr.models import EncDecClassificationModel
import copy
import resampy

# vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained('vad_marblenet')
# cfg = copy.deepcopy(vad_model._cfg)
# vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)
# vad_model.eval()
# vad_model = vad_model.to(vad_model.device)

# class AudioDataLayer(IterableDataset):
#     @property
#     def output_types(self):
#         return {
#             'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
#             'a_sig_length': NeuralType(tuple('B'), LengthsType()),
#         }

#     def __init__(self, sample_rate):
#         super().__init__()
#         self._sample_rate = sample_rate
#         self.output = True
        
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         if not self.output:
#             raise StopIteration
#         self.output = False
#         return torch.as_tensor(self.signal, dtype=torch.float32), \
#                torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
#     def set_signal(self, signal):
#         self.signal = signal.astype(np.float32)/32768.
#         self.signal_shape = self.signal.size
#         self.output = True

#     def __len__(self):
#         return 1
    
# data_layer = AudioDataLayer(sample_rate=cfg.train_ds.sample_rate)
# data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)

# def infer_signal(model, signal):
#     data_layer.set_signal(signal)
#     batch = next(iter(data_loader))
#     audio_signal, audio_signal_len = batch
#     audio_signal, audio_signal_len = audio_signal.to(vad_model.device), audio_signal_len.to(vad_model.device)
#     logits = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
#     return logits

# class FrameVAD:
    
#     def __init__(self, model_definition,
#                  threshold=0.5,
#                  frame_len=2, frame_overlap=2.5, 
#                  offset=10):
#         '''
#         Args:
#           threshold: If prob of speech is larger than threshold, classify the segment to be speech.
#           frame_len: frame's duration, seconds
#           frame_overlap: duration of overlaps before and after current frame, seconds
#           offset: number of symbols to drop for smooth streaming
#         '''
#         self.vocab = list(model_definition['labels'])
#         self.vocab.append('_')
        
#         self.sr = model_definition['sample_rate']
#         self.threshold = threshold
#         self.frame_len = frame_len
#         self.n_frame_len = int(frame_len * self.sr)
#         self.frame_overlap = frame_overlap
#         self.n_frame_overlap = int(frame_overlap * self.sr)
#         timestep_duration = model_definition['AudioToMFCCPreprocessor']['window_stride']
#         for block in model_definition['JasperEncoder']['jasper']:
#             timestep_duration *= block['stride'][0] ** block['repeat']
#         self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
#                                dtype=np.float32)
#         self.offset = offset
#         self.reset()
        
#     def _decode(self, frame, offset=0):
#         assert len(frame)==self.n_frame_len
#         self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
#         self.buffer[-self.n_frame_len:] = frame
#         logits = infer_signal(vad_model, self.buffer).cpu().numpy()[0]
#         decoded = self._greedy_decoder(
#             self.threshold,
#             logits,
#             self.vocab
#         )
#         return decoded  
    
    
#     @torch.no_grad()
#     def transcribe(self, frame=None):
#         if frame is None:
#             frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
#         if len(frame) < self.n_frame_len:
#             frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
#         unmerged = self._decode(frame, self.offset)
#         return unmerged
    
#     def reset(self):
#         '''
#         Reset frame_history and decoder's state
#         '''
#         self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
#         self.prev_char = ''

#     @staticmethod
#     def _greedy_decoder(threshold, logits, vocab):
#         s = []
#         if logits.shape[0]:
#             probs = torch.softmax(torch.as_tensor(logits), dim=-1)
#             probas, _ = torch.max(probs, dim=-1)
#             probas_s = probs[1].item()
#             preds = 1 if probas_s >= threshold else 0
#             s = [preds, str(vocab[preds]), probs[0].item(), probs[1].item(), str(logits)]
#         return s
    

# asr_model = None

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Loading the Tortoise model...")
#     global asr_model
#     if asr_model is None:
#         print("Loading the Tortoise model...")
#         asr_model = nemo_asr.models.ASRModel.from_pretrained(
#             # "nvidia/parakeet-ctc-1.1b"
#             "nvidia/stt_en_fastconformer_ctc_large"
#             # "nvidia/stt_en_citrinet_1024_gamma_0_25"
#             )
#         #nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_large")
#     yield

vad_model = EncDecClassificationModel.from_pretrained(model_name="vad_marblenet")

app = FastAPI(
    #lifespan=lifespan
              )


# @app.get('/ping')
# def health_check():
#     return {"status": asr_model is not None}



@app.post("/vad")
async def vad_endpoint(request: Request
                       #file: UploadFile = File(...)
                       ):
    # # Read audio data
    # audio_data, sample_rate = sf.read(file.file, dtype='float32')

    # # Resample to model's expected sample rate (16kHz if necessary)
    # target_sample_rate = 16000
    # if sample_rate != target_sample_rate:
    #     audio_data = resampy.resample(audio_data, sample_rate, target_sample_rate)

    data = await request.body()
    audio_data = np.frombuffer(data, dtype=np.float32)
    frameSize=12000 if request.headers['frame'] is None else int(request.headers['frame'])
    thres=0.6 if request.headers['thres'] is None else float(request.headers['thres'])

    # Perform VAD in chunks if necessary
    audio_chunks = np.array_split(audio_data, len(audio_data) // frameSize)  # 0.75-second chunks
    voice_segments = []
    prob_list=[]
    for chunk in audio_chunks:
        inputs = torch.tensor(chunk).unsqueeze(0).to(vad_model.device)  # Add batch dimension
        with torch.no_grad():
            logits = vad_model.forward(input_signal=inputs, input_signal_length=torch.tensor([len(chunk)]))
            probs = torch.softmax(logits[0], dim=-1)
            prob_list.append(probs[1].item())
            voice_activity = probs[1].item() > thres  # threshold for VAD
            voice_segments.append(voice_activity)

    return {"voice_segments": voice_segments,"probs":prob_list}

# @app.post("/transcribe")
# async def transcribe_audio(request: Request#file: UploadFile = File(...), 
#                         #    repeat:int = Form(...)
#                            ):
#     try:
        
#         data = await request.body()
#         audio_data = np.frombuffer(data, dtype=np.float32)

#         global asr_model
#         audio_data = torch.as_tensor(np.repeat([audio_data], repeats=repeat, axis=0) if repeat!=0 else [audio_data], 
#                                      dtype=torch.float32).to(asr_model.device)

#         # Prepare the signal length (batch size is 1, so it's a single value)
#         input_signal_length = torch.tensor(np.repeat([audio_data.shape[1]], repeats=repeat, axis=0) if repeat!=0 else [audio_data.shape[1]], 
#                                            dtype=torch.int64).to(asr_model.device)
        
#         preds = []
#         proba_b = []
#         proba_s = []
#         while len(data) > 0:

#         data = wf.readframes(CHUNK_SIZE)
#         signal = np.frombuffer(data, dtype=np.int16)
#         result = vad.transcribe(signal)

#         preds.append(result[0])
#         proba_b.append(result[2])
#         proba_s.append(result[3])
        
#         if len(result):
#             print(result,end='\n')
#             empty_counter = 3
#         elif empty_counter > 0:
#             empty_counter -= 1
#             if empty_counter == 0:
#                 print(' ',end='')
                
#     p.terminate()
#     vad.reset()
    
#     return preds, proba_b, proba_s
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
