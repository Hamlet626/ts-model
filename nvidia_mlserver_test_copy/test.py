import base64

# Specify the path to your audio file
audio_file_path = "kis_color_sky_short16000.wav"
output_file_path = "base64_audio.txt"

# Open the audio file in binary mode
with open(audio_file_path, "rb") as audio_file:
    # Read the audio file's binary content
    audio_data = audio_file.read()

    # Encode the binary content to Base64
    base64_encoded_audio = base64.b64encode(audio_data).decode('utf-8')

    with open(output_file_path, "w") as output_file:
        output_file.write(base64_encoded_audio)
