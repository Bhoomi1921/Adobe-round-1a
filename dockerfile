# FROM python:3.10.14-slim-bullseye

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get upgrade -y && apt-get install -y \
#     build-essential \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first for better caching
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Create necessary directories
# RUN mkdir -p documents results model/tokenizer

# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV TOKENIZERS_PARALLELISM=false

# # Expose Streamlit port
# EXPOSE 8501

# # Health check
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# # Default command
# CMD ["streamlit", "run", "streamlit_app_enhanced.py", "--server.port=8501", "--server.address=0.0.0.0"]
# Use slim Python image with 3.12.10
# Use slim Python image with 3.12.10
FROM python:3.12.10

WORKDIR /app

# Install system dependencies with cleanup in single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages with timeout and retries
RUN pip install --default-timeout=100 --retries 5 --no-cache-dir -r requirements.txt

# Copy application code (excluding unnecessary files)
COPY . .

# Create necessary directories
RUN mkdir -p documents results model/tokenizer

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

# Clean up unnecessary files
RUN rm -f streamlit_app_enhanced.py && \
    find . -type d -name "__pycache__" -exec rm -rf {} + && \
    find . -type f -name "*.pyc" -delete

# Set default command to run your main script
CMD ["python", "main.py"]