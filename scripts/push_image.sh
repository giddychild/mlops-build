#!/bin/bash
#!/bin/bash

set -e

# permissions for other scripts
chmod a+x ./scripts/banner.sh
chmod a+x ./scripts/auth.sh
chmod a+x ./scripts/app_changes.sh

# export variables
export PATH="$BITBUCKET_CLONE_DIR:$PATH"
export PATH="$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/bin/tput"
BASEDIR=$PWD
ENV=$1

./scripts/auth.sh

# Getting list of app changes
./scripts/app_changes.sh $BITBUCKET_COMMIT

push()
{
  APP_NAME=$1
  DOCKER_IMAGE=./tmp-image-$APP_NAME.docker

  ./scripts/banner.sh "TASK STARTED - $APP_NAME : Pushing Docker Image... --> "
  docker load --input $DOCKER_IMAGE

  TRGT_REPO=gcr.io/$PROJECT/$APP_NAME
  TRGT_IMG_TAG="${TAG:-latest}"

  ./scripts/banner.sh " Image available in local env..."
  docker images

  echo "ls current dir..."
  pwd && ls -l && chmod a+x *

  ./scripts/banner.sh "$APP_NAME - Pushing image to GCR repo.. "

  echo "docker push  $TRGT_REPO:$TRGT_IMG_TAG"

  docker push  $TRGT_REPO:$TRGT_IMG_TAG

  echo "image has been pushed into google registry..."

  ./scripts/banner.sh " <--- TASK COMPLETED - $APP_NAME : Pushing Docker Image into Google Registry... "

}

# Checking if an app is changed
if [ $(wc -l < "files_changed.txt") -eq 0 ]; then
    exit 0
else
    while read a; do
      push $a
    done <files_changed.txt
fi