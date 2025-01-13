# Use NVIDIA's PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:23.05-py3

# Install system dependencies
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for the app
WORKDIR /app

# Copy the requirements.txt first
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI app
CMD ["mlserver", "start", "."]
