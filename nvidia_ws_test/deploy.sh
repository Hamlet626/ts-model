export ARTIFACT_CONTAINER=tortoise-tts
export ARTIFACT=vad
export CLOUD_RUN=vad

gcloud builds submit --tag us-central1-docker.pkg.dev/sparko-473c5/${ARTIFACT_CONTAINER}/${ARTIFACT} --machine-type e2-highcpu-32

gcloud beta run deploy ${CLOUD_RUN} --port 8080 --region=us-central1 --image us-central1-docker.pkg.dev/sparko-473c5/${ARTIFACT_CONTAINER}/${ARTIFACT} --quiet