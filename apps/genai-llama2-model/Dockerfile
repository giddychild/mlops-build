#Fetching base image from gcr or ecr using prefix
ARG CONTAINER_REGISTRY_PREFIX=gcr.io/ca-mlops/
FROM ${CONTAINER_REGISTRY_PREFIX}gpu-pytorch

# Setting environment variables
ENV APP_HOME=/app

ENV PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
ENV CUDA_HOME=/usr/local/cuda

ENV GITHUB_ACTIONS=true

# Setting work directory
WORKDIR $APP_HOME

# Copying requirements.txt
#COPY ./apps/aws-llama-bot/requirements.txt ./
COPY requirements.txt ./
#COPY models/llama2_7b/ models/llama2_7b/  View GENAI-20 ticket

# Installing pip packages
RUN pip install --no-cache-dir -r requirements.txt

RUN pip3 uninstall -y auto-gptq torch torchvision torchaudio && \
    pip3 install --no-cache-dir auto-gptq==0.2.2

# Adding files to the container
#COPY ./apps/aws-llama-bot/src/ ./
COPY src/ ./
#COPY ./apps/aws-llama-bot/html/ ./templates
#COPY html/ ./templates
# Exposing port
EXPOSE 8080

# Running command
CMD [ "python3", "-m" , "flask", "--app", "server", "run", "--host=0.0.0.0", "--port=8080"]
