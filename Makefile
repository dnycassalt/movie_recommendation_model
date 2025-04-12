# Variables
PROJECT_ID ?= movie-recommendation-456203
REGION ?= us-east1
SERVICE_NAME ?= recommendation-service
MODEL_SOURCE ?= api/model.pth
MODEL_OUTPUT ?= api/model_prepared.pt
PYTHON ?= python3
FRONTEND_DIR ?= frontend
API_DIR ?= api
PORT ?= 5001
DOCKER_PORT ?= 8080

# Required files
REQUIRED_FILES := \
	$(API_DIR)/app.py \
	$(API_DIR)/model_prepared.pt \
	requirements.txt \
	data/processed/movies.csv \
	data/processed/users.csv \
	$(FRONTEND_DIR)/package.json

# Phony targets
.PHONY: auth init-gcloud build-frontend run-dev run-frontend dev build-prod push-prod deploy-prod prepare build deploy test clean help check-files run-docker

# Check required files
check-files:
	@echo "Checking required files..."
	@for file in $(REQUIRED_FILES); do \
		if [ ! -f "$$file" ]; then \
			echo "Error: Required file $$file not found"; \
			exit 1; \
		fi; \
	done
	@echo "All required files found"

# Authentication and initialization
auth: init-gcloud
	@echo "Authenticating with Google Cloud..."
	gcloud auth application-default login

init-gcloud:
	@echo "Initializing Google Cloud SDK..."
	gcloud init
	gcloud config set project $(PROJECT_ID)
	gcloud services enable run.googleapis.com
	gcloud services enable cloudbuild.googleapis.com

# Development
dev: check-files
	cd $(API_DIR) && python app.py

run-dev: check-files
	@echo "Starting Flask development server..."
	@echo "API will be available at http://localhost:$(PORT)"
	@echo "Press Ctrl+C to stop"
	cd $(API_DIR) && FLASK_ENV=development FLASK_DEBUG=1 PORT=$(PORT) $(PYTHON) run_dev.py

run-frontend: check-files
	@echo "Starting frontend development server..."
	@echo "Frontend will be available at http://localhost:3000"
	@echo "API URL: http://localhost:$(PORT)"
	cd $(FRONTEND_DIR) && REACT_APP_API_URL=http://localhost:$(PORT) npm start

# Docker
run-docker: check-files
	@echo "Cleaning up any existing containers..."
	docker stop app-local 2>/dev/null || true
	docker rm app-local 2>/dev/null || true
	docker rmi app-local 2>/dev/null || true
	
	@echo "Building frontend..."
	cd $(FRONTEND_DIR) && npm run build
	
	@echo "Building Docker image for linux/amd64..."
	docker buildx build --platform=linux/amd64 -t app-local .
	
	@echo "Starting container..."
	docker run --platform linux/amd64 \
		-p $(DOCKER_PORT):8080 \
		-e PORT=8080 \
		-e FLASK_ENV=production \
		--name app-local \
		app-local

# Frontend
build-frontend: check-files
	@echo "Building frontend..."
	@echo "Cleaning previous build..."
	cd $(FRONTEND_DIR) && rm -rf build node_modules/.cache
	@echo "Installing dependencies..."
	cd $(FRONTEND_DIR) && npm ci
	@echo "Building with cache-busting..."
	cd $(FRONTEND_DIR) && REACT_APP_API_URL=http://localhost:$(PORT) \
		GENERATE_SOURCEMAP=false \
		INLINE_RUNTIME_CHUNK=false \
		npm run build
	@echo "Frontend build completed successfully"

# Model preparation
prepare: check-files
	@echo "Preparing model for deployment..."
	cd $(API_DIR) && $(PYTHON) prepare_model.py --source $(MODEL_SOURCE) --output $(MODEL_OUTPUT)

# Production
build-prod: check-files
	@echo "Building production image..."
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "Error: PROJECT_ID is not set. Please run 'gcloud config set project YOUR_PROJECT_ID' or set PROJECT_ID environment variable."; \
		exit 1; \
	fi
	@echo "Building frontend with production environment..."
	cd $(FRONTEND_DIR) && npm run build
	@echo "Copying required files..."
	@for file in $(API_DIR)/model_prepared.pt data/processed/movies.csv data/processed/users.csv; do \
		if [ ! -f "$$file" ]; then \
			echo "Error: Required file $$file not found"; \
			exit 1; \
		fi; \
		cp "$$file" .; \
	done
	@echo "Building Docker image..."
	docker build -t gcr.io/$(PROJECT_ID)/$(SERVICE_NAME) .
	@echo "Cleaning up..."
	rm model_prepared.pt
	rm movies.csv
	rm users.csv

push-prod:
	@echo "Pushing to Google Container Registry..."
	docker push gcr.io/$(PROJECT_ID)/$(SERVICE_NAME)

deploy-prod:
	@echo "Deploying to Cloud Run..."
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "Error: PROJECT_ID is not set. Please run 'gcloud config set project YOUR_PROJECT_ID' or set PROJECT_ID environment variable."; \
		exit 1; \
	fi
	@if [ -z "$(REGION)" ]; then \
		echo "Error: REGION is not set. Please set REGION environment variable."; \
		exit 1; \
	fi
	@SERVICE_URL=$$(gcloud run deploy $(SERVICE_NAME) \
		--image gcr.io/$(PROJECT_ID)/$(SERVICE_NAME) \
		--platform managed \
		--region $(REGION) \
		--allow-unauthenticated \
		--memory 2Gi \
		--cpu 2 \
		--min-instances 0 \
		--max-instances 10 \
		--concurrency 80 \
		--format='value(status.url)') && \
	echo "Service URL: $$SERVICE_URL" && \
	sed -i '' "s|REACT_APP_API_URL=.*|REACT_APP_API_URL=$$SERVICE_URL|" $(FRONTEND_DIR)/.env.production

# Testing and cleanup
test: check-files
	@echo "Testing API locally..."
	docker run -p 8080:8080 gcr.io/$(PROJECT_ID)/$(SERVICE_NAME)

clean:
	@echo "Cleaning up..."
	rm -f $(MODEL_OUTPUT)
	rm -rf $(FRONTEND_DIR)/build

# Help
help:
	@echo "Available commands:"
	@echo ""
	@echo "Local Development:"
	@echo "  make dev            - Run the Flask development server directly"
	@echo "  make run-dev        - Run Flask in development mode with debug enabled"
	@echo "  make run-frontend   - Run frontend in development mode (separate process)"
	@echo "  make run-docker     - Run the full app in Docker (API + frontend)"
	@echo ""
	@echo "Development Setup:"
	@echo "  make check-files    - Check if all required files exist"
	@echo "  make auth           - Authenticate with Google Cloud"
	@echo "  make init-gcloud    - Initialize Google Cloud SDK"
	@echo ""
	@echo "Build and Deploy:"
	@echo "  make build-frontend - Build the frontend application"
	@echo "  make prepare        - Prepare the model for deployment"
	@echo "  make build-prod     - Build the production Docker image"
	@echo "  make push-prod      - Push the image to Google Container Registry"
	@echo "  make deploy-prod    - Deploy to Cloud Run and update frontend URL"
	@echo ""
	@echo "Testing and Maintenance:"
	@echo "  make test           - Test the API locally using Docker"
	@echo "  make clean          - Clean up temporary files"
	@echo ""
	@echo "Local Development Guide:"
	@echo "  1. Quick Start:"
	@echo "     $$ make run-docker"
	@echo "     This runs the full app (API + frontend) in Docker on port $(DOCKER_PORT)"
	@echo ""
	@echo "  2. Development Mode:"
	@echo "     Terminal 1: $$ make run-dev"
	@echo "     Terminal 2: $$ make run-frontend"
	@echo "     This runs the API and frontend separately for easier debugging"
	@echo ""
	@echo "  3. Direct Flask:"
	@echo "     $$ make dev"
	@echo "     Runs Flask directly without Docker"
	@echo ""
	@echo "Configuration:"
	@echo "  PROJECT_ID     - Google Cloud project ID (default: $(PROJECT_ID))"
	@echo "  REGION         - Cloud Run region (default: $(REGION))"
	@echo "  SERVICE_NAME   - Cloud Run service name (default: $(SERVICE_NAME))"
	@echo "  MODEL_SOURCE   - Source model path (default: $(MODEL_SOURCE))"
	@echo "  MODEL_OUTPUT   - Output model path (default: $(MODEL_OUTPUT))"
	@echo "  PYTHON         - Python executable (default: $(PYTHON))"
	@echo "  FRONTEND_DIR   - Frontend directory (default: $(FRONTEND_DIR))"
	@echo "  API_DIR        - API directory (default: $(API_DIR))"
	@echo "  PORT           - Development server port (default: $(PORT))"
	@echo "  DOCKER_PORT    - Docker container port (default: $(DOCKER_PORT))"
	@echo ""
	@echo "Required files:"
	@for file in $(REQUIRED_FILES); do \
		echo "  $$file"; \
	done 