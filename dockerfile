# # Use slim Python image on amd64 architecture
# FROM --platform=linux/amd64 python:3.12.10-slim

# # Set the working directory
# WORKDIR /app

# # Copy requirements file and install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code
# COPY . .

# # Ensure required folders exist
# RUN mkdir -p documents results model/tokenizer

# # Set the default command to run your main processing script
# CMD ["python", "main.py"]
# Use slim Python image
FROM python:3.12.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create required folders
RUN mkdir -p documents results model/tokenizer

# Default command
CMD ["python", "main.py"]

# FROM --platform=linux/amd64 python:3.10-slim  # Changed to more stable Python 3.10

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONPATH=/app

# # Set the working directory
# WORKDIR /app

# # Install system dependencies first (if needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     # Add any system packages your app needs here
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements file and install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code
# COPY . .

# # Create directories with correct permissions
# RUN mkdir -p input results

# # Set the default command to run your main processing script
# ENTRYPOINT ["python", "main.py"]