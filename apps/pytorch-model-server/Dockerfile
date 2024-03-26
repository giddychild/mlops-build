FROM ubuntu:20.04

# common environemnt variables
ENV NB_USER jovyan
ENV NB_UID 1000
ENV NB_PREFIX /
ENV HOME /home/$NB_USER
ENV SHELL /bin/bash
ENV ROOT_PASS $ROOT_PASS
ENV PATH="${PATH}:/home/jovyan/.local/bin"

# set shell to bash
SHELL ["/bin/bash", "-c"]

# install - usefull linux packages
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get -yq update \
 && apt-get -yq install --no-install-recommends \
    apt-transport-https \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    default-jre \
    gcc \
    git \
    gnupg \
    gnupg2 \
    libkrb5-dev \
    locales \
    lsb-release \
    nano \
    software-properties-common \
    tzdata \
    unzip \
    vim \
    wget \
    zip \
    build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# create user and set required ownership
RUN useradd -M -s /bin/bash -N -u ${NB_UID} ${NB_USER} -g users \
 && mkdir -p ${HOME} \
 && chown -R ${NB_USER}:users ${HOME} \
 && chown -R ${NB_USER}:users /usr/local/bin

# set locale configs
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
 && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

USER root

# install -- node.js and python
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get -yq update \
 && apt-get -yq install --no-install-recommends \
    python3.10\
    python3-pip\
    python3-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

# switch to NB_UID for installs
USER ${NB_UID}

# install requirements.txt and Scala kernel
COPY --chown=jovyan:users requirements.txt /tmp
RUN python3 -m pip install --upgrade -r /tmp/requirements.txt --no-cache-dir \
 && rm -f /tmp/requirements.txt