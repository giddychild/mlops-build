FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
# Copying requirements.txt
COPY requirements.txt ./
# Installing pip packages
RUN pip install --no-cache-dir -r requirements.txt
# Adding files to the container
COPY src/ ./
COPY .streamlit .streamlit
# Exposing port
EXPOSE 80
HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health
# Running command
CMD [ "streamlit", "run" , "app.py", "--server.port=80", "--server.address=0.0.0.0"]