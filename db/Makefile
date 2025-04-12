# Variables
PROJECT_ID ?= $(shell gcloud config get-value project)
DB_INSTANCE_NAME ?= movie-recommendation-db
DB_NAME ?= movie_recommendation
DB_USER ?= app-user
DB_PASSWORD ?= $(shell openssl rand -base64 12)
ROOT_PASSWORD ?= $(shell openssl rand -base64 12)
REGION ?= us-central1
TIER ?= db-f1-micro
STORAGE_SIZE ?= 10GB

.PHONY: init-db create-db-user create-service-account setup-db setup-secrets create-schema help preprocess-csv import-data

help:
	@echo "Available commands:"
	@echo "  make init-db              - Initialize Cloud SQL instance"
	@echo "  make create-db-user       - Create database user"
	@echo "  make create-service-account - Create service account"
	@echo "  make setup-secrets        - Set up secrets in Secret Manager"
	@echo "  make create-schema        - Create database schema"
	@echo "  make setup-db             - Run all setup steps"
	@echo "  make preprocess-csv       - Preprocess CSV files"
	@echo "  make import-data          - Import data into the database"

init-db:
	@echo "Enabling required APIs..."
	gcloud services enable sqladmin.googleapis.com
	gcloud services enable compute.googleapis.com

	@echo "Creating Cloud SQL instance..."
	gcloud sql instances create $(DB_INSTANCE_NAME) \
		--database-version=POSTGRES_14 \
		--tier=$(TIER) \
		--region=$(REGION) \
		--root-password=$(ROOT_PASSWORD) \
		--storage-size=$(STORAGE_SIZE) \
		--storage-type=SSD \
		--backup-start-time=02:00 \
		--maintenance-window-day=SAT \
		--maintenance-window-hour=02

	@echo "Creating database..."
	gcloud sql databases create $(DB_NAME) \
		--instance=$(DB_INSTANCE_NAME)

	@echo "Cloud SQL instance created successfully!"
	@echo "Instance name: $(DB_INSTANCE_NAME)"
	@echo "Database name: $(DB_NAME)"
	@echo "Root password: $(ROOT_PASSWORD)"

create-db-user:
	@echo "Creating database user..."
	gcloud sql users create $(DB_USER) \
		--instance=$(DB_INSTANCE_NAME) \
		--password=$(DB_PASSWORD)
	@echo "Database user created successfully!"
	@echo "Username: $(DB_USER)"
	@echo "Password: $(DB_PASSWORD)"

create-service-account:
	@echo "Creating service account..."
	gcloud iam service-accounts create movie-recommendation-sa \
		--description="Service account for movie recommendation app" \
		--display-name="Movie Recommendation SA"

	@echo "Granting permissions..."
	gcloud projects add-iam-policy-binding $(PROJECT_ID) \
		--member="serviceAccount:movie-recommendation-sa@$(PROJECT_ID).iam.gserviceaccount.com" \
		--role="roles/cloudsql.client"

	@echo "Creating service account key..."
	gcloud iam service-accounts keys create service-account-key.json \
		--iam-account=movie-recommendation-sa@$(PROJECT_ID).iam.gserviceaccount.com

	@echo "Service account created successfully!"
	@echo "Key file: service-account-key.json"

setup-secrets:
	@echo "Enabling Secret Manager API..."
	gcloud services enable secretmanager.googleapis.com

	@echo "Creating secrets..."
	@echo -n "$(DB_PASSWORD)" | gcloud secrets create db-password \
		--data-file=- \
		--replication-policy="automatic"

	@echo -n "$(DB_USER)" | gcloud secrets create db-username \
		--data-file=- \
		--replication-policy="automatic"

	@echo -n "$(DB_INSTANCE_NAME)" | gcloud secrets create db-instance-name \
		--data-file=- \
		--replication-policy="automatic"

	@echo -n "$(DB_NAME)" | gcloud secrets create db-name \
		--data-file=- \
		--replication-policy="automatic"

	@echo "Granting Secret Manager access to service account..."
	gcloud secrets add-iam-policy-binding db-password \
		--member="serviceAccount:movie-recommendation-sa@$(PROJECT_ID).iam.gserviceaccount.com" \
		--role="roles/secretmanager.secretAccessor"

	gcloud secrets add-iam-policy-binding db-username \
		--member="serviceAccount:movie-recommendation-sa@$(PROJECT_ID).iam.gserviceaccount.com" \
		--role="roles/secretmanager.secretAccessor"

	gcloud secrets add-iam-policy-binding db-instance-name \
		--member="serviceAccount:movie-recommendation-sa@$(PROJECT_ID).iam.gserviceaccount.com" \
		--role="roles/secretmanager.secretAccessor"

	gcloud secrets add-iam-policy-binding db-name \
		--member="serviceAccount:movie-recommendation-sa@$(PROJECT_ID).iam.gserviceaccount.com" \
		--role="roles/secretmanager.secretAccessor"

	@echo "Secrets created successfully!"

create-schema:
	@echo "Creating database schema..."
	@cat > schema.sql << EOF
	CREATE TABLE IF NOT EXISTS users (
		_id VARCHAR,
		display_name VARCHAR,
		num_ratings_pages INTEGER,
		num_reviews INTEGER,
		username VARCHAR PRIMARY KEY
	);

	CREATE TABLE IF NOT EXISTS movies (
		_id VARCHAR,
		genres VARCHAR DEFAULT 'Unknown',
		image_url VARCHAR DEFAULT '',
		imdb_id VARCHAR DEFAULT '',
		imdb_link VARCHAR DEFAULT '',
		movie_id VARCHAR PRIMARY KEY,
		movie_title VARCHAR DEFAULT 'Untitled',
		original_language VARCHAR DEFAULT 'en',
		overview TEXT DEFAULT '',
		popularity FLOAT DEFAULT 0.0,
		production_countries VARCHAR DEFAULT 'Unknown',
		release_date DATE DEFAULT '1970-01-01',
		runtime INTEGER DEFAULT 0,
		spoken_languages VARCHAR DEFAULT 'English',
		tmdb_id VARCHAR DEFAULT '',
		tmdb_link VARCHAR DEFAULT '',
		vote_average FLOAT DEFAULT 0.0,
		vote_count INTEGER DEFAULT 0,
		year_released INTEGER DEFAULT 1970
	);

	CREATE TABLE IF NOT EXISTS ratings (
		_id VARCHAR,
		movie_id VARCHAR,
		rating_val FLOAT CHECK (rating_val >= 0 AND rating_val <= 10),
		user_id VARCHAR,
		PRIMARY KEY (user_id, movie_id)
	);

	CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id);
	CREATE INDEX IF NOT EXISTS idx_ratings_movie_id ON ratings(movie_id);
	EOF

	@echo "Creating temporary bucket for schema import..."
	@BUCKET_NAME=temp-schema-$(shell date +%s)
	@gsutil mb -l $(REGION) gs://$(BUCKET_NAME)

	@echo "Uploading schema to bucket..."
	@gsutil cp schema.sql gs://$(BUCKET_NAME)/

	@echo "Importing schema to database..."
	@gcloud sql import sql $(DB_INSTANCE_NAME) \
		gs://$(BUCKET_NAME)/schema.sql \
		--database=$(DB_NAME) \
		--quiet

	@echo "Cleaning up..."
	@gsutil rm gs://$(BUCKET_NAME)/schema.sql
	@gsutil rb gs://$(BUCKET_NAME)
	@rm schema.sql

	@echo "Database schema created successfully!"

setup-db: init-db create-db-user create-service-account setup-secrets create-schema
	@echo "Database setup completed successfully!"
	@echo "Please save these credentials securely:"
	@echo "Root password: $(ROOT_PASSWORD)"
	@echo "App user password: $(DB_PASSWORD)"
	@echo "Service account key: service-account-key.json"

preprocess-csv:
	@echo "Preprocessing CSV files..."
	@python scripts/preprocess_csv.py

import-data: preprocess-csv
	@echo "Creating temporary bucket for data import..."
	@BUCKET_NAME=temp-import-$(shell date +%s)
	@gsutil mb -l $(REGION) gs://$(BUCKET_NAME)

	@echo "Uploading processed files to bucket..."
	@gsutil cp data/processed/movies.csv gs://$(BUCKET_NAME)/
	@gsutil cp data/processed/ratings.csv gs://$(BUCKET_NAME)/
	@gsutil cp data/processed/users.csv gs://$(BUCKET_NAME)/

	@echo "Importing movies data..."
	@gcloud sql import csv $(DB_INSTANCE_NAME) \
		gs://$(BUCKET_NAME)/movies.csv \
		--database=$(DB_NAME) \
		--table=movies \
		--columns=_id,genres,image_url,imdb_id,imdb_link,movie_id,movie_title,original_language,overview,popularity,production_countries,release_date,runtime,spoken_languages,tmdb_id,tmdb_link,vote_average,vote_count,year_released

	@echo "Importing users data..."
	@gcloud sql import csv $(DB_INSTANCE_NAME) \
		gs://$(BUCKET_NAME)/users.csv \
		--database=$(DB_NAME) \
		--table=users \
		--columns=_id,display_name,num_ratings_pages,num_reviews,username

	@echo "Importing ratings data..."
	@gcloud sql import csv $(DB_INSTANCE_NAME) \
		gs://$(BUCKET_NAME)/ratings.csv \
		--database=$(DB_NAME) \
		--table=ratings \
		--columns=_id,movie_id,rating_val,user_id

	@echo "Cleaning up..."
	@gsutil rm gs://$(BUCKET_NAME)/*.csv
	@gsutil rb gs://$(BUCKET_NAME)
	@echo "Data import completed successfully!" 