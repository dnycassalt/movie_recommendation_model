# Use Python 3.8 slim image as base
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Create and activate virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model file
COPY --chown=appuser:appuser model_prepared.pt .

# Copy the application code
COPY --chown=appuser:appuser app.py .

# Copy the frontend build
COPY --chown=appuser:appuser ../frontend/build /app/frontend

# Install production server
RUN pip install gunicorn

# Set environment variables
ENV PORT=8080
ENV MODEL_PATH=/app/model_prepared.pt

# Configure nginx
COPY --chown=appuser:appuser nginx.conf /etc/nginx/nginx.conf

# Configure supervisor
COPY --chown=appuser:appuser supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Switch back to root for supervisor
USER root

# Run supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 