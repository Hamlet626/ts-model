from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset
import torchaudio
import soundfile as sf
import torch

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.generation_config.language = "english"


# Load local audio file
filename = "test.wav"
# waveform, sample_rate = torchaudio.load(filename, format="wav")
waveform, sample_rate = sf.read(filename)
waveform = torch.tensor(waveform, dtype=torch.float32)

target_sample_rate = 16000
if sample_rate != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    waveform = resampler(waveform)
    
# waveform = torch.tensor(waveform).unsqueeze(0)


# Preprocess the audio to get input features
input_features = processor(waveform.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt", language='en').input_features

# Generate token ids
predicted_ids = model.generate(input_features)

# Decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)


# # load dummy dataset and read audio files
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]
# input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# # generate token ids
# predicted_ids = model.generate(input_features)
# # decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
