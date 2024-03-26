FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Setting environment variables
ENV APP_HOME=/app

# Setting work directory
WORKDIR $APP_HOME

# Copying requirements.txt
# COPY requirements.txt ./

RUN apt-get update -y \
    && apt-get install -y python3-pip

# Installing pip packages
RUN pip install kserve

RUN pip install vllm accelerate fschat

# Adding files to the container
COPY src/ ./

# Exposing port
EXPOSE 8080

# Running command
CMD [ "python3", "model.py"]
