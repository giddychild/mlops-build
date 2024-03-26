FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS vllm-openai

# common environemnt variables
ENV NB_USER vllm
ENV NB_UID 1000
ENV HOME /home/$NB_USER
ENV PATH="${PATH}:/home/${NB_USER}/.local/bin"

# create user and set required ownership
RUN useradd -M -s /bin/bash -N -u ${NB_UID} ${NB_USER} -g users -G sudo \
 && mkdir -p ${HOME} \
 && chown -R ${NB_USER}:users ${HOME} \
 && chown -R ${NB_USER}:users /usr/local/bin

USER root

RUN apt-get update -y \
    && apt-get install -y python3-pip

USER ${NB_UID}

RUN pip install vllm accelerate fschat

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]