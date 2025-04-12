# Database Operations

## Data Import/Export

### Importing Data

1. Import users:
   ```bash
   psql -h $DB_INSTANCE_NAME -U $DB_USER -d $DB_NAME -c "\copy users(user_id, username) FROM 'users.csv' WITH CSV HEADER"
   ```

2. Import movies:
   ```bash
   psql -h $DB_INSTANCE_NAME -U $DB_USER -d $DB_NAME -c "\copy movies(movie_id, title, year) FROM 'movies.csv' WITH CSV HEADER"
   ```

3. Import ratings:
   ```bash
   psql -h $DB_INSTANCE_NAME -U $DB_USER -d $DB_NAME -c "\copy ratings(user_id, movie_id, rating) FROM 'ratings.csv' WITH CSV HEADER"
   ```

### Exporting Data

1. Export users:
   ```bash
   psql -h $DB_INSTANCE_NAME -U $DB_USER -d $DB_NAME -c "\copy users TO 'users.csv' WITH CSV HEADER"
   ```

2. Export movies:
   ```bash
   psql -h $DB_INSTANCE_NAME -U $DB_USER -d $DB_NAME -c "\copy movies TO 'movies.csv' WITH CSV HEADER"
   ```

3. Export ratings:
   ```bash
   psql -h $DB_INSTANCE_NAME -U $DB_USER -d $DB_NAME -c "\copy ratings TO 'ratings.csv' WITH CSV HEADER"
   ```

## Common Queries

### User Operations

1. Get user details:
   ```sql
   SELECT * FROM users WHERE user_id = :user_id;
   ```

2. Get user's ratings:
   ```sql
   SELECT m.title, r.rating, r.timestamp 
   FROM ratings r 
   JOIN movies m ON r.movie_id = m.movie_id 
   WHERE r.user_id = :user_id 
   ORDER BY r.timestamp DESC;
   ```

### Movie Operations

1. Get movie details:
   ```sql
   SELECT * FROM movies WHERE movie_id = :movie_id;
   ```

2. Get movie ratings:
   ```sql
   SELECT u.username, r.rating, r.timestamp 
   FROM ratings r 
   JOIN users u ON r.user_id = u.user_id 
   WHERE r.movie_id = :movie_id 
   ORDER BY r.timestamp DESC;
   ```

### Rating Operations

1. Add a rating:
   ```sql
   INSERT INTO ratings (user_id, movie_id, rating) 
   VALUES (:user_id, :movie_id, :rating);
   ```

2. Update a rating:
   ```sql
   UPDATE ratings 
   SET rating = :rating, timestamp = CURRENT_TIMESTAMP 
   WHERE user_id = :user_id AND movie_id = :movie_id;
   ```

3. Delete a rating:
   ```sql
   DELETE FROM ratings 
   WHERE user_id = :user_id AND movie_id = :movie_id;
   ```

## Maintenance

### Backup

1. Create a backup:
   ```bash
   gcloud sql backups create --instance=$DB_INSTANCE_NAME
   ```

2. List backups:
   ```bash
   gcloud sql backups list --instance=$DB_INSTANCE_NAME
   ```

3. Restore from backup:
   ```bash
   gcloud sql backups restore --instance=$DB_INSTANCE_NAME --backup-id=$BACKUP_ID
   ```

### Monitoring

1. Check database size:
   ```sql
   SELECT pg_size_pretty(pg_database_size('$DB_NAME'));
   ```

2. Check table sizes:
   ```sql
   SELECT table_name, pg_size_pretty(pg_total_relation_size(table_name))
   FROM information_schema.tables
   WHERE table_schema = 'public'
   ORDER BY pg_total_relation_size(table_name) DESC;
   ```

3. Check index usage:
   ```sql
   SELECT schemaname, relname, idx_scan, idx_tup_read, idx_tup_fetch
   FROM pg_stat_user_indexes;
   ```

## Performance Optimization

### Index Maintenance

1. Analyze tables:
   ```sql
   ANALYZE users;
   ANALYZE movies;
   ANALYZE ratings;
   ```

2. Vacuum tables:
   ```sql
   VACUUM ANALYZE users;
   VACUUM ANALYZE movies;
   VACUUM ANALYZE ratings;
   ```

### Query Optimization

1. Use EXPLAIN to analyze query plans:
   ```sql
   EXPLAIN ANALYZE SELECT * FROM ratings WHERE user_id = :user_id;
   ```

2. Check for missing indexes:
   ```sql
   SELECT schemaname, tablename, indexname, indexdef
   FROM pg_indexes
   WHERE schemaname = 'public';
   ```

## Security

### User Management

1. Create a new user:
   ```sql
   CREATE USER new_user WITH PASSWORD 'password';
   ```

2. Grant permissions:
   ```sql
   GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO new_user;
   ```

3. Revoke permissions:
   ```sql
   REVOKE ALL ON ALL TABLES IN SCHEMA public FROM new_user;
   ```

### Audit Logging

1. Enable audit logging:
   ```sql
   ALTER DATABASE $DB_NAME SET log_statement = 'all';
   ```

2. Check audit logs:
   ```bash
   gcloud logging read "resource.type=cloudsql_database AND resource.labels.database_id=$DB_INSTANCE_NAME" --limit=50
   ``` 