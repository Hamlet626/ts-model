
# gcloud builds submit --tag us-central1-docker.pkg.dev/sparko-473c5/tortoise-tts/tortoise-ml-server --machine-type e2-highcpu-32

# docker buildx build --platform linux/amd64 -t 'mlserver_test' .
# # mlserver build . -t 'mlserver_test'

# docker tag mlserver_test us-central1-docker.pkg.dev/sparko-473c5/tortoise-tts/tortoise-ml-server:mlserver

# docker push us-central1-docker.pkg.dev/sparko-473c5/tortoise-tts/tortoise-ml-server:mlserver

# gcloud beta run deploy tts-test --region=us-central1 --image us-central1-docker.pkg.dev/sparko-473c5/tortoise-tts/tortoise-ml-server:mlserver --quiet

gcloud builds submit --tag us-central1-docker.pkg.dev/sparko-473c5/tortoise-tts/tortoise-server --machine-type e2-highcpu-32

gcloud beta run deploy tts-test --region=us-central1 --image us-central1-docker.pkg.dev/sparko-473c5/tortoise-tts/tortoise-server --quiet