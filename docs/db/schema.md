# Database Schema

## Overview

The movie recommendation system uses a PostgreSQL database with the following tables:

- `users`: Stores user information
- `movies`: Stores movie information
- `ratings`: Stores user ratings for movies

## Tables

### Users Table

```sql
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Movies Table

```sql
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Ratings Table

```sql
CREATE TABLE ratings (
    user_id INTEGER REFERENCES users(user_id),
    movie_id INTEGER REFERENCES movies(movie_id),
    rating DECIMAL(3,1) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, movie_id)
);

CREATE INDEX idx_ratings_user_id ON ratings(user_id);
CREATE INDEX idx_ratings_movie_id ON ratings(movie_id);
```

## Indexes

The following indexes are created to improve query performance:

1. `idx_ratings_user_id`: For fast lookups of user ratings
2. `idx_ratings_movie_id`: For fast lookups of movie ratings

## Constraints

- Primary keys ensure unique identification of users and movies
- Foreign key constraints maintain referential integrity between tables
- Rating values are stored as decimal numbers with one decimal place
- Timestamps are automatically set to the current time when records are created 