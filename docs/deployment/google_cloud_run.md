# Google Cloud Run Deployment Guide

This guide provides step-by-step instructions for deploying your PyTorch recommendation model to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**
   - Create a project in Google Cloud Console
   - Enable billing for your project
   - Note your Project ID

2. **Local Setup**
   - Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
   - Install Docker
   - Install Python 3.8 or later
   - Have a trained model file ready

3. **Authentication**
   ```bash
   # Initialize gcloud
   gcloud init
   
   # Set your project
   gcloud config set project [PROJECT_ID]
   
   # Enable necessary APIs
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

## Automated Deployment

The deployment process is automated using a Makefile in the `api` directory.

### 1. Configure Deployment

Edit the `api/Makefile` to set your configuration:
```makefile
PROJECT_ID = your-project-id
REGION = us-central1
SERVICE_NAME = recommendation-service
MODEL_SOURCE = ../models/best_model.pth
MODEL_OUTPUT = ../model.pt
```

### 2. Deploy the Service

Run the deployment commands:
```bash
# Navigate to the API directory
cd api

# Deploy the service
make prepare build deploy
```

This will:
1. Prepare the model for deployment by:
   - Loading the model from the source path
   - Optimizing it for production
   - Saving it to the output path
2. Build the Docker container using the local Dockerfile
3. Deploy to Cloud Run

### 3. Verify Deployment

Check the service status:
```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe recommendation-service \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)')

# Test the API
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 1}'
```

## Manual Deployment Steps

If you prefer to deploy manually, follow these steps:

### 1. Prepare the Model

```bash
# Prepare the model
python prepare_model.py --source /path/to/model.pth --output model.pt
```

### 2. Build the Container

```bash
# Navigate to the API directory
cd api

# Build the container
docker build -t recommendation-model .

# Test locally
docker run -p 8080:8080 recommendation-model
```

### 3. Deploy to Cloud Run

```bash
# Build and push the container
gcloud builds submit --tag gcr.io/[PROJECT_ID]/recommendation-model .

# Deploy the service
gcloud run deploy recommendation-service \
  --image gcr.io/[PROJECT_ID]/recommendation-model \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Configuration Options

### Auto-scaling

```bash
# Set minimum and maximum instances
gcloud run services update recommendation-service \
  --min-instances=1 \
  --max-instances=10 \
  --cpu-throttling
```

### Memory Allocation

```bash
# Configure memory
gcloud run services update recommendation-service \
  --memory=512Mi
```

## Monitoring and Logging

### View Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=recommendation-service" --limit 50
```

### View Metrics

- Open [Cloud Console](https://console.cloud.google.com)
- Navigate to Cloud Run > recommendation-service
- View metrics dashboard

## Troubleshooting

1. **Common Issues**
   - Container fails to start: Check logs for initialization errors
   - High latency: Consider increasing memory allocation
   - Cold starts: Set minimum instances to 1

2. **Debugging Steps**
   ```bash
   # View recent logs
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=recommendation-service" --limit 50
   
   # Check service status
   gcloud run services describe recommendation-service --region us-central1
   
   # View build logs
   gcloud builds log [BUILD_ID]
   ```

## Cost Optimization

1. **Resource Allocation**
   - Start with 512Mi memory
   - Monitor usage and adjust as needed
   - Use CPU throttling for cost savings

2. **Scaling Configuration**
   - Set appropriate min/max instances
   - Use concurrency settings
   - Monitor and adjust based on usage patterns

## Security Best Practices

1. **Authentication**
   ```bash
   # Enable authentication
   gcloud run services update recommendation-service \
     --no-allow-unauthenticated
   ```

2. **Service Account**
   ```bash
   # Create service account
   gcloud iam service-accounts create recommendation-service
   
   # Grant necessary permissions
   gcloud projects add-iam-policy-binding [PROJECT_ID] \
     --member="serviceAccount:recommendation-service@[PROJECT_ID].iam.gserviceaccount.com" \
     --role="roles/run.invoker"
   ```

## Next Steps

1. **Performance Testing**
   - Use load testing tools
   - Monitor response times
   - Adjust resources as needed

2. **Feature Updates**
   - Implement batch predictions
   - Add caching layer
   - Set up monitoring alerts

3. **Documentation**
   - Create API documentation
   - Document deployment process
   - Maintain runbook for operations 