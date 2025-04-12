# Database Setup

## Prerequisites

- Google Cloud SDK installed and configured
- Project ID set in environment variable `PROJECT_ID`
- Cloud SQL API enabled
- Secret Manager API enabled

## Setup Steps

1. Create Cloud SQL instance:
   ```bash
   make create-instance
   ```

2. Create database and user:
   ```bash
   make create-db
   ```

3. Create service account:
   ```bash
   make create-service-account
   ```

4. Set up secrets:
   ```bash
   make setup-secrets
   ```

5. Create database schema:
   ```bash
   make create-schema
   ```

Or run all steps at once:
```bash
make setup-db
```

## Environment Variables

The following environment variables are required:

- `PROJECT_ID`: Your Google Cloud project ID
- `DB_INSTANCE_NAME`: Name of the Cloud SQL instance
- `DB_NAME`: Name of the database
- `DB_USER`: Database username
- `DB_PASSWORD`: Database password

## Local Development

For local development:

1. Set up application default credentials:
   ```bash
   gcloud auth application-default login
   ```

2. Export project ID:
   ```bash
   export PROJECT_ID=your-project-id
   ```

3. Run the setup commands as described above

## Troubleshooting

### Common Issues

1. **Permission Denied**
   - Ensure the service account has the necessary permissions
   - Check that the Secret Manager API is enabled
   - Verify that the service account has access to the secrets

2. **Connection Issues**
   - Verify that the Cloud SQL instance is running
   - Check that the IP address is whitelisted
   - Ensure the database credentials are correct

3. **Schema Creation Failed**
   - Check that the database exists
   - Verify that the user has sufficient privileges
   - Ensure the SQL commands are valid

### Getting Help

If you encounter any issues:

1. Check the Google Cloud Console for error messages
2. Review the setup logs
3. Contact support with the error details 