import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_ctc_large")

# Export the model to ONNX
asr_model.export('model.onnx')
