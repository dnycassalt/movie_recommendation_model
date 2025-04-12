# Cloud Deployment Guide

This guide covers deploying your trained PyTorch recommendation model to either Google Cloud or AWS.

## Prerequisites

- Trained PyTorch model (saved as `.pth` file)
- Docker installed locally
- Cloud provider account (Google Cloud or AWS)
- Cloud CLI tools installed:
  - Google Cloud SDK for GCP
  - AWS CLI for AWS

## Option 1: Google Cloud Deployment

### 1. Prepare Model for Deployment

```python
# save_for_deployment.py
import torch

def prepare_model_for_deployment(model_path, output_path):
    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    
    # Export to TorchScript
    example_users = torch.tensor([1, 2, 3])
    example_movies = torch.tensor([1, 2, 3])
    traced_model = torch.jit.trace(model, (example_users, example_movies))
    
    # Save the traced model
    traced_model.save(output_path)
```

### 2. Create Docker Container

```dockerfile
# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and application files
COPY ./model.pt .
COPY ./app.py .

# Install production server
RUN pip install gunicorn

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
```

### 3. Create Flask Application

```python
# app.py
from flask import Flask, request, jsonify
import torch
import json

app = Flask(__name__)

# Load the model
model = torch.jit.load('model.pt')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = torch.tensor([data['user_id']])
    movie_id = torch.tensor([data['movie_id']])
    
    with torch.no_grad():
        prediction = model(user_id, movie_id)
    
    return jsonify({'rating': prediction.item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### 4. Deploy to Google Cloud Run

```bash
# Build and push the container
gcloud builds submit --tag gcr.io/[PROJECT_ID]/recommendation-model

# Deploy to Cloud Run
gcloud run deploy recommendation-service \
  --image gcr.io/[PROJECT_ID]/recommendation-model \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Option 2: AWS Deployment

### 1. Prepare Model (same as above)

### 2. Create Docker Container (same as above)

### 3. Deploy to AWS Elastic Container Service (ECS)

```bash
# Build and tag the Docker image
docker build -t recommendation-model .

# Create ECR repository
aws ecr create-repository --repository-name recommendation-model

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [AWS_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com

# Tag and push the image
docker tag recommendation-model:latest [AWS_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/recommendation-model:latest
docker push [AWS_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/recommendation-model:latest
```

### 4. Create ECS Task Definition

```json
{
  "family": "recommendation-service",
  "containerDefinitions": [
    {
      "name": "recommendation-model",
      "image": "[AWS_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/recommendation-model:latest",
      "memory": 512,
      "cpu": 256,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "hostPort": 8080,
          "protocol": "tcp"
        }
      ]
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "memory": "512",
  "cpu": "256"
}
```

## Performance Optimization

### 1. Model Optimization

```python
def optimize_model_for_production(model):
    # Quantize the model
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Export optimized model
    return quantized_model
```

### 2. Batch Prediction Support

```python
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.json
    user_ids = torch.tensor(data['user_ids'])
    movie_ids = torch.tensor(data['movie_ids'])
    
    with torch.no_grad():
        predictions = model(user_ids, movie_ids)
    
    return jsonify({'ratings': predictions.tolist()})
```

## Monitoring and Scaling

### Google Cloud Monitoring

```python
from opencensus.ext.stackdriver import stats_exporter
from opencensus.stats import stats

# Set up monitoring
stats_exporter = stats_exporter.new_stats_exporter()
stats.stats.view_manager.register_exporter(stats_exporter)

def log_prediction_metrics(prediction_time, batch_size):
    stats.stats.record([
        ('prediction_latency', prediction_time),
        ('batch_size', batch_size)
    ])
```

### AWS CloudWatch

```python
import boto3
import time

cloudwatch = boto3.client('cloudwatch')

def log_metrics(prediction_time, batch_size):
    cloudwatch.put_metric_data(
        Namespace='RecommendationService',
        MetricData=[
            {
                'MetricName': 'PredictionLatency',
                'Value': prediction_time,
                'Unit': 'Milliseconds'
            },
            {
                'MetricName': 'BatchSize',
                'Value': batch_size,
                'Unit': 'Count'
            }
        ]
    )
```

## Best Practices

1. **Model Versioning**
   - Use semantic versioning for models
   - Keep model metadata with version information
   - Implement A/B testing capability

2. **Error Handling**
   ```python
   @app.errorhandler(Exception)
   def handle_error(error):
       return jsonify({
           'error': str(error),
           'status': 'error'
       }), 500
   ```

3. **Health Checks**
   ```python
   @app.route('/health', methods=['GET'])
   def health_check():
       return jsonify({
           'status': 'healthy',
           'model_version': MODEL_VERSION
       })
   ```

4. **Load Testing**
   ```python
   # locustfile.py
   from locust import HttpUser, task, between

   class RecommendationUser(HttpUser):
       wait_time = between(1, 3)

       @task
       def predict_rating(self):
           self.client.post("/predict", json={
               "user_id": 1,
               "movie_id": 1
           })
   ```

## Cost Optimization

1. **Auto-scaling Configuration**

For Google Cloud Run:
```bash
gcloud run services update recommendation-service \
    --min-instances=1 \
    --max-instances=10 \
    --cpu-throttling
```

For AWS ECS:
```json
{
  "targetValue": 75.0,
  "scaleOutCooldown": 300,
  "scaleInCooldown": 300,
  "predefinedMetricSpecification": {
    "predefinedMetricType": "ECSServiceAverageCPUUtilization"
  }
}
```

2. **Resource Optimization**
   - Use appropriate instance sizes
   - Implement caching for frequent predictions
   - Monitor and adjust based on usage patterns

## Security Considerations

1. **API Authentication**
   ```python
   from functools import wraps
   
   def require_api_key(f):
       @wraps(f)
       def decorated_function(*args, **kwargs):
           api_key = request.headers.get('X-API-Key')
           if not api_key or not validate_api_key(api_key):
               return jsonify({'error': 'Invalid API key'}), 401
           return f(*args, **kwargs)
       return decorated_function
   ```

2. **Input Validation**
   ```python
   def validate_input(user_id, movie_id):
       if not isinstance(user_id, int) or not isinstance(movie_id, int):
           raise ValueError("Invalid input types")
       if user_id < 0 or movie_id < 0:
           raise ValueError("Invalid input values")
   ```

## Continuous Deployment

### Google Cloud Build

```yaml
# cloudbuild.yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/recommendation-model', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/recommendation-model']
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  entrypoint: gcloud
  args:
  - 'run'
  - 'deploy'
  - 'recommendation-service'
  - '--image'
  - 'gcr.io/$PROJECT_ID/recommendation-model'
  - '--region'
  - 'us-central1'
  - '--platform'
  - 'managed'
```

### AWS CodePipeline

```yaml
# buildspec.yml
version: 0.2

phases:
  pre_build:
    commands:
      - aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
  build:
    commands:
      - docker build -t recommendation-model .
      - docker tag recommendation-model:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/recommendation-model:latest
  post_build:
    commands:
      - docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/recommendation-model:latest
      - printf '[{"name":"recommendation-model","imageUri":"%s"}]' $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/recommendation-model:latest > imagedefinitions.json
``` 