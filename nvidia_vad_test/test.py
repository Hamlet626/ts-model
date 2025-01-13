import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-ctc-0.6b")

transcriptions = asr_model.transcribe(["file.wav"])