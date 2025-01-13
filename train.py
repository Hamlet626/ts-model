from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import time

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

t1=time.time()
speech = synthesiser("The sky is usually blue during the day because of the way the sunlight shines through the air. But sometimes it can be orange, pink, or gray too, especially when the sun is setting or when it’s cloudy!", forward_params={"speaker_embeddings": speaker_embedding})

print(time.time()-t1)
sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

