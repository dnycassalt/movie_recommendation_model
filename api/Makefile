# Variables
PROJECT_ID ?= movie-recommendation-456203
REGION ?= us-east1
SERVICE_NAME ?= recommendation-service
MODEL_SOURCE ?= ./model.pth
MODEL_OUTPUT ?= ./model_prepared.pt
PYTHON ?= python3
FRONTEND_DIR ?= ../frontend
PORT ?= 5001

# Phony targets
.PHONY: prepare build deploy test clean help auth init-gcloud build-frontend run-dev run-frontend

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

# Run Flask in development mode
run-dev:
	@echo "Starting Flask development server..."
	@echo "API will be available at http://localhost:$(PORT)"
	@echo "Press Ctrl+C to stop"
	FLASK_ENV=development FLASK_DEBUG=1 PORT=$(PORT) $(PYTHON) run_dev.py

# Run frontend in development mode
run-frontend:
	@echo "Starting frontend development server..."
	@echo "Frontend will be available at http://localhost:3000"
	@echo "API URL: http://localhost:$(PORT)"
	cd $(FRONTEND_DIR) && REACT_APP_API_URL=http://localhost:$(PORT) npm start

# Build frontend
build-frontend:
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

# Prepare the model for deployment
prepare:
	@echo "Preparing model for deployment..."
	$(PYTHON) ../prepare_model.py --source $(MODEL_SOURCE) --output $(MODEL_OUTPUT)

# Build the Docker container
build: build-frontend prepare
	@echo "Building Docker container..."
	docker build -t recommendation-model .

# Deploy to Cloud Run
deploy:
	@echo "Deploying to Cloud Run..."
	gcloud builds submit --tag gcr.io/$(PROJECT_ID)/recommendation-model .
	gcloud run deploy $(SERVICE_NAME) \
		--image gcr.io/$(PROJECT_ID)/recommendation-model \
		--platform managed \
		--region $(REGION) 

# Test the API locally
test:
	@echo "Testing API locally..."
	docker run -p 8080:8080 recommendation-model

# Clean up temporary files
clean:
	@echo "Cleaning up..."
	rm -f $(MODEL_OUTPUT)
	rm -rf $(FRONTEND_DIR)/build

# Help message
help:
	@echo "Available commands:"
	@echo "  make auth           - Authenticate with Google Cloud"
	@echo "  make init-gcloud    - Initialize Google Cloud SDK"
	@echo "  make build-frontend - Build the frontend application"
	@echo "  make prepare        - Prepare the model for deployment"
	@echo "  make build          - Build the Docker container"
	@echo "  make deploy         - Deploy to Cloud Run"
	@echo "  make test           - Test the API locally"
	@echo "  make run-dev        - Run Flask in development mode"
	@echo "  make run-frontend   - Run frontend in development mode"
	@echo "  make clean          - Clean up temporary files"
	@echo "  make help           - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  PROJECT_ID     - Google Cloud project ID (default: $(PROJECT_ID))"
	@echo "  REGION         - Cloud Run region (default: $(REGION))"
	@echo "  SERVICE_NAME   - Cloud Run service name (default: $(SERVICE_NAME))"
	@echo "  MODEL_SOURCE   - Source model path (default: $(MODEL_SOURCE))"
	@echo "  MODEL_OUTPUT   - Output model path (default: $(MODEL_OUTPUT))"
	@echo "  PYTHON         - Python executable (default: $(PYTHON))"
	@echo "  FRONTEND_DIR   - Frontend directory (default: $(FRONTEND_DIR))"
	@echo "  PORT           - Development server port (default: $(PORT))" 