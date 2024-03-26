#!/bin/bash

# Downloading model file from aws s3 bucket
mkdir ./apps/flight-call-center-aws/model

aws s3api get-object --bucket srcd-mlops-flight-center-model-ca-central-1 --key call_center_intent_model_v1_0.h5 ./apps/flight-call-center-aws/model/call_center_intent_model_v1_0.h5
aws s3api get-object --bucket srcd-mlops-flight-center-model-ca-central-1 --key tokenizer_v1_0.json ./apps/flight-call-center-aws/model/tokenizer_v1_0.json
