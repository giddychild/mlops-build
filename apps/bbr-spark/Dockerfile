# Fetching base image from gcr or ecr using prefix
ARG CONTAINER_REGISTRY_PREFIX=gcr.io/ca-mlops/
FROM ${CONTAINER_REGISTRY_PREFIX}spark-py:base_3.2.1

# Install kubectl
USER root
RUN apt-get update
RUN apt install -y apt-transport-https apt-utils gnupg curl \
  && curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | tee -a /etc/apt/sources.list.d/kubernetes.list \
  && apt update \
  && apt install -y kubectl

# Switch back to Spark USER uid
USER ${spark_uid}