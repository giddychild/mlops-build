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
# ./scripts/app_changes.sh $BITBUCKET_COMMIT

cat files_changed.txt

build()
{
  APP_NAME=$1
  TRGT_REPO=gcr.io/$PROJECT/$APP_NAME
  TRGT_IMG_TAG="${TAG:-latest}"

  gcloud container clusters get-credentials $CLUSTER_NAME --zone=$COMPUTE_ZONE

  kubectl cluster-info

  kubectl delete pod $APP_NAME-builder-$BITBUCKET_BUILD_NUMBER || true

  sleep 5    

  ./scripts/banner.sh "TASK STARTED - $APP_NAME: Building Docker Image... --> "

  kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $APP_NAME-builder-$BITBUCKET_BUILD_NUMBER
  namespace: default
spec:
  containers:
    - name: kaniko
      image: gcr.io/kaniko-project/executor:latest
      resources:
        requests:
          memory: "4096Mi"
          cpu: "2000m"
        limits:
          memory: "16000Mi"
          cpu: "6000m"
      args:
        - "--dockerfile=Dockerfile"
        - "--cleanup"
        - "--context=gs://srcd-mlops-dev-datasets/$APP_NAME/$APP_NAME.tar.gz"
        - "--destination=$TRGT_REPO:$TRGT_IMG_TAG"
        - "--compressed-caching=false"
        - "--use-new-run"         
      volumeMounts:
        - name: kaniko-secret
          mountPath: /secret
      env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /secret/srcd-mlops-dev.bitbucket-runner-363.json
  restartPolicy: Never
  volumes:
    - name: kaniko-secret
      secret:
        secretName: kaniko-secret
EOF

  sleep 5
  
  gcloud container images list-tags $TRGT_REPO

  kubectl logs -f $APP_NAME-builder-$BITBUCKET_BUILD_NUMBER kaniko

  kubectl delete pod $APP_NAME-builder-$BITBUCKET_BUILD_NUMBER || true

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

