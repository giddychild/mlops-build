FROM python:3.10-slim

ENV APP_HOME=/app

#Setting up work directory
WORKDIR $APP_HOME


#Copying requirements.txt and installing pip packages
COPY ./apps/automl-experiment/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

#Adding files to the container
COPY ./apps/automl-experiment/src/automl_experiment.py ./
# COPY ./apps/automl-experiment/glove.6B.100d.txt ./
# COPY ./apps/automl-experiment/call_center_intents.csv ./

CMD ["python3", "-m", "automl_experiment"]