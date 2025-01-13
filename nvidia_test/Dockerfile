# Use NVIDIA's PyTorch image with CUDA support
# FROM nvcr.io/nvidia/pytorch:23.05-py3
FROM nvcr.io/nvidia/nemo:24.07

# Install system dependencies
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for the app
WORKDIR /app

# Copy the requirements.txt first
COPY requirements.txt /app/requirements.txt

# for vad endpoint, to prevent errot
# RuntimeError: operator torchvision::nms does not exist
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
