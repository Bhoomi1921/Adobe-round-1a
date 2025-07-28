# Use official Python slim image for AMD64 architecture
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app
# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY document_parser.py .
COPY hierarchy_builder.py .
COPY content_classifier.py .
COPY semantic_analyzer.py .
COPY main.py .

# Create input and output directories
RUN mkdir -p documents results model/tokenizer

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command to run the application

CMD ["python", "main.py"]