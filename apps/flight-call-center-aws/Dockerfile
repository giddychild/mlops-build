#Docker file for AWS
FROM python:3.8.10-slim

ENV PATH="/app:${PATH}"
ENV APP_HOME=/app
ENV FLASK_APP=server.py

# Setting work directory
WORKDIR $APP_HOME

# Copying requirements.txt
COPY ./apps/flight-call-center-aws/requirements.txt ./

# Installing pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Adding files to the container
COPY ./apps/flight-call-center-aws/src/ ./apps/flight-call-center-aws/model/ ./
COPY ./apps/flight-call-center-aws/html/ ./templates

# Exposing port
EXPOSE 8080

# Making serve executable
RUN chmod +x serve