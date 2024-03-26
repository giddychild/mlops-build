#!/bin/bash

set -e

echo $SA_KEY | base64 -d > ./.sa_key.json
gcloud auth activate-service-account $SA --key-file ./.sa_key.json
gcloud auth configure-docker gcr.io 
rm -f ./.sa_key.json
