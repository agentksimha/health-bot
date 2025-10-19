# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Create writable cache directories
RUN mkdir -p /app/.cache/huggingface/hub /app/.cache/torch
RUN chmod -R 777 /app/.cache

# Set environment variables for caches
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV TORCH_HOME=/app/.cache/torch
ENV XDG_CACHE_HOME=/app/.cache

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit with the modular demo
CMD ["streamlit", "run", "demo_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
