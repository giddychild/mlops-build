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

build()
{
  APP_NAME=$1

  ./scripts/banner.sh "TASK STARTED - $APP_NAME: Packaging Files... --> "

  mkdir -p ./apps/Packages/$APP_NAME

  tar -cvzf $APP_NAME.tar.gz -C ./apps/$APP_NAME/ ./

  cp $APP_NAME.tar.gz ./apps/Packages/$APP_NAME/

  sleep 5  

  ./scripts/banner.sh "TASK STARTED - $APP_NAME: Uploading Files... --> "

  gsutil cp ./apps/Packages/$APP_NAME/$APP_NAME.tar.gz gs://srcd-mlops-dev-datasets/$APP_NAME/$APP_NAME.tar.gz

  sleep 5  

  rm -r ./apps/Packages

  ./scripts/banner.sh " <--- TASK COMPLETED- $APP_NAME : Packaging Files... "  
}

# Checking if an app is changed
if [ $(wc -l < "files_changed.txt") -eq 0 ]; then
    exit 0
else
    while read a; do
      build $a
    done <files_changed.txt
fi

