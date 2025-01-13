
gcloud builds submit --tag us-central1-docker.pkg.dev/sparko-473c5/tortoise-tts/tortoise-server --machine-type e2-highcpu-32

gcloud beta run deploy tts-test --port 8080 --region=us-central1 --image us-central1-docker.pkg.dev/sparko-473c5/tortoise-tts/tortoise-server --quiet