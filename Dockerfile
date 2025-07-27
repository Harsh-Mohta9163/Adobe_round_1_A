# round1a/Dockerfile

# Use the same base image you provided
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install system dependencies (from your original file)
RUN apt-get update && apt-get install -y \
    build-essential tesseract-ocr poppler-utils libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt averaged_perceptron_tagger

# Copy all your application code
COPY ./app ./app
COPY complete_pipeline.py .
COPY docker_runner.py .

# The command to run when the container starts
CMD ["python", "docker_runner.py"]