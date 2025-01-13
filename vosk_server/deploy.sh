
# gcloud builds submit --tag us-central1-docker.pkg.dev/sparko-473c5/tortoise-tts/vosk-server --machine-type e2-highcpu-32

gcloud beta run deploy tts-test --region=us-central1 --image us-central1-docker.pkg.dev/sparko-473c5/tortoise-tts/vosk-server --port 2700 --quiet