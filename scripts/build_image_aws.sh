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

build()
{
  APP_NAME=$1

  DOCKER_FILE=./apps/$APP_NAME/Dockerfile

  INIT_FILE=./apps/$APP_NAME/init/init.sh

  if [[ -f "$INIT_FILE" ]]; then
    chmod a+x "$INIT_FILE"
    . "$INIT_FILE"
  fi


  ./scripts/banner.sh "TASK STARTED - $APP_NAME: Building Docker Image... --> "

  # Getting source image repo, name and tag from Dockerfile
  SRC_IMG_NAME=`awk '$1=="FROM" { print $2 }' $DOCKER_FILE`
  TRGT_REPO=426746725987.dkr.ecr.ca-central-1.amazonaws.com/flight-call-center
  TRGT_IMG_TAG="${TAG:-latest}"

  aws ecr get-login-password --region ca-central-1 | docker login --username AWS --password-stdin $TRGT_REPO
  
  ./scripts/banner.sh "List local docker image..."
  docker images

  ./scripts/banner.sh "pull base image from source repo..."
  echo "docker pull $SRC_IMG_NAME"
  docker pull $SRC_IMG_NAME

  echo "ls current dir..."
  pwd && ls -l

  ./scripts/banner.sh "start baking docker image.. "
  chmod a+x *

  echo "docker build -t $TRGT_REPO:$TRGT_IMG_TAG . -f $DOCKER_FILE"

  docker build -t $TRGT_REPO:$TRGT_IMG_TAG . -f $DOCKER_FILE 
  docker save --output tmp-image-$APP_NAME.docker  $TRGT_REPO:$TRGT_IMG_TAG

  echo "image has been built..."
  docker images -a 

  ./scripts/banner.sh " <--- TASK COMPLETED- $APP_NAME : Building Docker Image... "
  
}

# Checking if an app is changed
if [ $(wc -l < "files_changed.txt") -eq 0 ]; then
    exit 0
else
    while read a; do
      build $a
    done <files_changed.txt
fi

