# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and source code
COPY models/ ./models/
COPY src/ ./src/
COPY tests/ ./tests/

# Set Python path
ENV PYTHONPATH=/app

# Run tests to verify everything works
RUN python -m pytest tests/test_model.py -v

# Command to run when container starts
CMD ["python", "-c", "print('✅ ML Model container is ready!'); from src.model import load_model; model = load_model('models/churn_model.pkl'); print('✅ Model loaded successfully!')"]