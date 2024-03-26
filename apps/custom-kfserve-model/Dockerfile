FROM python:3.7-slim

ENV APP_HOME=/app

# Setting work directory
WORKDIR $APP_HOME

# Copying files in the image
COPY ./apps/custom-kfserve-model/app.py ./
COPY ./apps/custom-kfserve-model/requirements.txt ./

# Installing dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["python", "app.py"]