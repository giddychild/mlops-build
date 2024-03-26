#!/bin/bash

set -e

# permissions for other scripts
chmod a+x ./scripts/banner.sh
chmod a+x ./scripts/app_changes.sh

# export variables
export PATH="$BITBUCKET_CLONE_DIR:$PATH"
export PATH="$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/bin/tput"
BASEDIR=$PWD
ENV=$1

# Getting list of app changes
./scripts/app_changes.sh $BITBUCKET_COMMIT

scan()
{
  APP_NAME=$1
  DOCKER_IMAGE=./tmp-image-$APP_NAME.docker

  ./scripts/banner.sh "TASK START - $APP_NAME : Scanning Image Vulnerability... --> "

  # Loading the docker image created in the build step
  docker load --input $DOCKER_IMAGE

  ./scripts/banner.sh "List local docker image..."
  docker images

  TRGT_REPO=gcr.io/$PROJECT/$APP_NAME
  TRGT_IMG_TAG="${TAG:-latest}"

  # Running Trivy to scan the docker image
  docker run aquasec/trivy image $TRGT_REPO:$TRGT_IMG_TAG --severity HIGH,CRITICAL

  if [[ $? -eq 0 ]]; then
  ./scripts/banner.sh " *****   SUCCESS- $APP_NAME : No Vulnerabilities Found ***** "  
    exit 0
  else
    ./scripts/banner.sh " *****  FAILED- $APP_NAME : Found Image Vulnerabilities ***** " 
    exit 1 
  fi
  
  ./scripts/banner.sh "TASK END- $APP_NAME : Scanning Image Vulnerability... --> "
}

# Checking if an app is changed
if [ $(wc -l < "files_changed.txt") -eq 0 ]; then
    exit 0
else
    while read a; do
      scan $a
    done <files_changed.txt
fi

