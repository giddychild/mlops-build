FROM python:3.8.10-slim

ENV APP_HOME=/app

# Setting work directory
WORKDIR $APP_HOME

# Copying requirements.txt
COPY ./apps/vertex-ai-poc/requirements.txt ./

# Installing pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Adding files to the container
COPY ./apps/vertex-ai-poc/src/ ./apps/vertex-ai-poc/model/ ./
COPY ./apps/vertex-ai-poc/html/ ./templates

# Exposing port
EXPOSE 8080

# Running command
CMD [ "python3", "-m" , "flask", "--app", "server", "run", "--host=0.0.0.0", "--port=8080"]
