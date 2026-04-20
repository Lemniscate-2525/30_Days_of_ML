# Using official Python lightweight image
FROM python:3.10-slim

# Setting working directory inside the container
WORKDIR /app

# Copying the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying entire project into the container
COPY . .

# Exposing the API and Frontend ports
EXPOSE 8000 8501
