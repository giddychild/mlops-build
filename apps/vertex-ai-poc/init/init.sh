#!/bin/bash

# Downloading model file from google storage
mkdir ./apps/vertex-ai-poc/model

gcloud storage cp gs://srcd-vertexai-model-bkt/flight-call-center/call_center_intent_model_v1.0.h5 ./apps/vertex-ai-poc/model/
gcloud storage cp gs://srcd-vertexai-model-bkt/flight-call-center/tokenizer_v1.0.json ./apps/vertex-ai-poc/model/