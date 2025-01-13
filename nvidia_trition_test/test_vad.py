import requests
import numpy as np
import requests
import soundfile as sf

audio_data, sample_rate = sf.read('kis_color_sky_short16000.wav', dtype='float32')

audio_data=np.array(audio_data.tolist()).astype(np.float32).tobytes()

response = requests.post('https://tts-test-74416368455.us-central1.run.app/vad',
                          data=audio_data,
                          headers={
                                   # frame is the time length for each element in returned response array, 
                                   # in which the api will split the entire audio_data into frame chunks and return result for each chunk,
                                   #  e.g. 16000 means each frame is 1 sec, since audio_data should be in 16000Hz, 
                                   #       12000 means each frame is 0.75 sec,
                                   # optional, default value is 12000
                                   'frame':'12000', 

                                   # thres is the threshold of determine a frame probability is background or speach.
                                   # This only affects response['voice_segments'], and currently the probs array is also returned for evaluation.
                                   #  e.g. 0.6 means >0.6 will be classified as speach, <=0.6 will be background noise.
                                   # optional, default value is 0.6
                                   'thres':'0.55'
                                   })

print(response.json())

data=response.json()['probs']

