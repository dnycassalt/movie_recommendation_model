# Final stage
FROM python:3.9-slim

WORKDIR /app

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY api/app.py .
COPY api/model_prepared.pt .
COPY data/processed/movies.csv .
COPY data/processed/users.csv .

# Copy the built frontend files
COPY frontend/build ./static

# Set environment variables
ENV PORT=8080
ENV MODEL_PATH=model_prepared.pt

# Expose port
EXPOSE 8080

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"] 