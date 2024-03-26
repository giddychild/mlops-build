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
# ./scripts/app_changes.sh $BITBUCKET_COMMIT

TRGT_REPO=426746725987.dkr.ecr.ca-central-1.amazonaws.com/flight-call-center
aws ecr get-login-password --region ca-central-1 | docker login --username AWS --password-stdin $TRGT_REPO

push()
{
  APP_NAME=$1
  DOCKER_IMAGE=./tmp-image-$APP_NAME.docker

  ./scripts/banner.sh "TASK STARTED - $APP_NAME : Pushing Docker Image... --> "
  docker load --input $DOCKER_IMAGE

  TRGT_IMG_TAG="${TAG:-latest}"

  ./scripts/banner.sh " Image available in local env..."
  docker images

  echo "ls current dir..."
  pwd && ls -l && chmod a+x *

  ./scripts/banner.sh "$APP_NAME - Pushing image to ECR repo.. "

  echo "docker push  $TRGT_REPO:$TRGT_IMG_TAG"

  docker push  $TRGT_REPO:$TRGT_IMG_TAG

  echo "image has been pushed into AWS registry..."

  ./scripts/banner.sh " <--- TASK COMPLETED - $APP_NAME : Pushing Docker Image into AWS Registry... "

}

# Checking if an app is changed
if [ $(wc -l < "files_changed.txt") -eq 0 ]; then
    exit 0
else
    while read a; do
      push $a
    done <files_changed.txt
fi