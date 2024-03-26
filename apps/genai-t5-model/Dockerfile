FROM python:3.9-slim

# Setting environment variables
ENV APP_HOME=/app

# Setting work directory
WORKDIR $APP_HOME

# Copying requirements.txt
COPY requirements.txt ./

# Installing pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Adding files to the container
COPY src/ ./

# Exposing port
EXPOSE 8080

# Running command
# CMD [ "python3", "-m" , "flask", "--app", "server", "run", "--host=0.0.0.0", "--port=8080"]
CMD [ "python", "model.py"]
