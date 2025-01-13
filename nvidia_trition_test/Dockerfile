# Use NVIDIA's Triton Inference Server image with PyTorch backend
FROM nvcr.io/nvidia/tritonserver:23.05-py3

ARG TRITON_ENABLE_PYTHON=ON
# Install any additional dependencies (if needed)
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for the model repository
WORKDIR /app

# Copy the requirements.txt first
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --upgrade cython \
    && pip install --no-cache-dir -r requirements.txt

# Copy your model repository into the container
COPY model-repo /app/model-repo

# ENV PORT=8000

# Expose the ports Triton uses
EXPOSE 8000 8001 8002

# Command to run Triton Inference Server
CMD ["tritonserver", "--model-repository=/app/model-repo"]
