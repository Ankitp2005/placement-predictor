# Placement Signal — Hugging Face Spaces Dockerfile
# Exposes the FastAPI app on port 7860 (HF Spaces default).

FROM python:3.12-slim

# Create a non-root user (HF Spaces best practice)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY model.pkl .
COPY shap_summary.png .
COPY index.html .

# Switch to non-root user
USER appuser

# Expose the port HF Spaces expects
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "app.py"]
