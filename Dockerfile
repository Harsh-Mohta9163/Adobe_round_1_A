# Use AMD64 compatible Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-jpn \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Download NLTK data
RUN python -m nltk.downloader punkt averaged_perceptron_tagger
# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p /app/input /app/output \
    /app/data/input /app/data/final_results \
    /app/data/md_files /app/data/spans_output /app/data/aggregator_output \
    /app/data/textlines_csv_output /app/data/textline_predictions \
    /app/data/merged_textblocks /app/data/textblock_predictions \
    /app/data/output_model1 /app/app/models_code/models

# Set Python path
ENV PYTHONPATH=/app:/app/app/extractor:/app/app/models_code:/app/app/merging

# Set the default command
CMD ["python", "/app/docker_runner.py"]