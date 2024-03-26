FROM python:3.7-slim

ENV APP_HOME=/app

# Setting work directory
WORKDIR $APP_HOME

# Installing pip packages
RUN pip install pandas scikit-learn flask gunicorn

# Adding files to the container
COPY ./apps/custom-model-serve/model ./model
COPY  ./apps/custom-model-serve/server.py server.py

# Exposing port
EXPOSE 8080

# Running command
CMD [ "gunicorn", "--bind", "0.0.0.0:8080", "server:app" ]