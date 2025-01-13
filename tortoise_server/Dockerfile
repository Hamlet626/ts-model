# # Base image
# FROM python:3.10-slim

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libsndfile1-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /app

# COPY . /app
# RUN pip install --upgrade pip
# RUN pip install -r /app/requirements.txt

# # Expose port for inference
# EXPOSE 8080

# # Command to run the FastAPI server
# ENTRYPOINT ["python", "server.py"]
# # CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]




# Use Ubuntu as the base image
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies, including Python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get install -y python3.10 python3-pip

# Upgrade pip to the latest version
# RUN python3 -m pip install --upgrade pip

# Copy the code into the image
COPY . /opt/code

# Install all dependencies from requirements.txt
RUN pip install --no-cache-dir -r /opt/code/requirements.txt
RUN python3 -m spacy download en_core_web_sm

# Set the working directory
WORKDIR /opt/code

EXPOSE 8080

# Entry point for running FastAPI server
# ENTRYPOINT ["python3", "server.py"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]




# # Stage 1: Builder
# FROM python:3.10-slim as builder

# # Set the working directory
# WORKDIR /app

# # Install system dependencies for building Python packages
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     python3-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements.txt and install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Stage 2: Runtime
# FROM python:3.10-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy Python dependencies from the builder stage
# COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
# COPY --from=builder /usr/local/bin /usr/local/bin
# COPY --from=builder /usr/local/include /usr/local/include

# # Copy the application code
# COPY . .

# # Expose the port FastAPI runs on
# EXPOSE 8080

# # Command to run the app with uvicorn
# CMD ["uvicorn", "tortoise_ser:app", "--host", "0.0.0.0", "--port", "8080"]
