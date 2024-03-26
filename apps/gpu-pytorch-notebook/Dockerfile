FROM ubuntu:20.04
ARG S6_OVERLAY_VERSION=3.1.5.0

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
    sudo \
    tzdata \
    unzip \
    vim \
    wget \
    xz-utils \
    zip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# install - s6 overlay
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz /tmp
RUN tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-x86_64.tar.xz /tmp
RUN tar -C / -Jxpf /tmp/s6-overlay-x86_64.tar.xz

# create user and set required ownership
RUN useradd -M -s /bin/bash -N -u ${NB_UID} ${NB_USER} -g users -G sudo \
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
 && curl -sL "https://deb.nodesource.com/gpgkey/nodesource.gpg.key" | apt-key add - \
 && echo "deb https://deb.nodesource.com/node_14.x focal main" > /etc/apt/sources.list.d/nodesource.list \
 && apt-get -yq update \
 && apt-get -yq install --no-install-recommends \
    nodejs \
    python3.8\
    python3-pip\
    python3-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip \
  && echo "%sudo	ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
  && chmod 777 /usr/local/share/

# install NVIDIA GPU drivers
COPY nvidia-drivers.sh /tmp/nvidia-drivers.sh
RUN sh /tmp/nvidia-drivers.sh

# switch to NB_UID for installs
USER ${NB_UID}

# install requirements.txt and Scala kernel
COPY --chown=jovyan:users requirements.txt /tmp
RUN python3 -m pip install --upgrade -r /tmp/requirements.txt --no-cache-dir \
 && rm -f /tmp/requirements.txt
COPY --chown=jovyan:users run-jupyter.sh /opt/run-jupyter.sh

# generate Jupyter config and install Spark kernel
RUN ${HOME}/.local/bin/jupyter lab --generate-config \
 && rm -rf ${HOME}/.cache/yarn \
 && chmod +x /opt/run-jupyter.sh

# Switch to root make copy of home directory
USER root

# s6 - copy scripts
COPY --chown=${NB_USER}:users --chmod=755 cont-init.d /etc/cont-init.d

RUN chmod +x /etc/cont-init.d/*

# s6 - 01-copy-tmp-home
RUN mkdir -p /tmp_home \
 && cp -r ${HOME} /tmp_home \
 && chown -R ${NB_USER}:users /tmp_home

 # switch to NB_UID for installs
USER ${NB_UID}

EXPOSE 8888

ENTRYPOINT ["/init"]