import os
import torch
import pandas as pd
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with correct static folder
app = Flask(__name__,
            static_folder='static',
            static_url_path='',
            template_folder='static'
            )

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",  # React development server
            "https://*.run.app",      # Cloud Run domains
            "http://localhost:5001"   # Local API
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Number of movies in the dataset
NUM_MOVIES = 286071

# Load user mapping
users_path = os.path.join('users.csv')
try:
    users_df = pd.read_csv(users_path)
    # Create mapping from username to numeric index
    username_to_id = {username: idx for idx,
                      username in enumerate(users_df['username'])}
    logger.info(f"Loaded {len(username_to_id)} users from {users_path}")

    # Create persona to username mapping
    persona_to_username = {
        'persona_1': 'filipe_furtado',  # 7,894 reviews
        'persona_2': 'settingsun',  # 7,121 reviews
        'persona_3': 'johntyler',  # 6,553 reviews
        'persona_4': 'colonelmortimer',  # 5,278 reviews
        'persona_5': 'zoltarak'  # 5,072 reviews
    }
    logger.info("Created persona to username mapping")
except Exception as e:
    logger.error(f"Error loading users data: {str(e)}")
    raise

# Load movie data
movies_path = os.path.join('movies.csv')
try:
    movies_df = pd.read_csv(movies_path)
    # Create mapping from numeric index to movie data
    movie_data = {idx: row for idx, row in movies_df.iterrows()}
    logger.info(f"Loaded {len(movie_data)} movies from {movies_path}")
except Exception as e:
    logger.error(f"Error loading movies data: {str(e)}")
    raise

# Load the model
model_path = os.getenv('MODEL_PATH', 'model_prepared.pt')
logger.info(f"Loading model from {model_path}")

try:
    model = torch.jit.load(model_path)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise


@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        persona = data.get('persona')

        # If persona is provided, use the mapped username
        if persona and persona in persona_to_username:
            username = persona_to_username[persona]
            logger.info(
                f"Using mapped username {username} for persona {persona}")

        # Convert username to numeric ID
        if username not in username_to_id:
            return jsonify({"error": f"User {username} not found"}), 404

        user_id = username_to_id[username]

        # Get predictions for all movies at once using vectorization
        movie_ids = torch.arange(NUM_MOVIES, dtype=torch.long)
        # Create a tensor of the same user ID repeated for all movies
        user_ids = torch.full((NUM_MOVIES,), user_id, dtype=torch.long)

        with torch.no_grad():
            # Make predictions for all movies in one batch
            predictions = model(user_ids, movie_ids)
            # Clamp all predictions between 0 and 10
            predictions = torch.clamp(predictions, 0, 10)
            # Convert to list of (movie_id, prediction) tuples
            predictions = list(zip(movie_ids.tolist(), predictions.tolist()))

        # Sort predictions and get top 50
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_50 = predictions[:50]

        # Convert to list format with movie details
        recommendations = []
        for movie_id, rating in top_50:
            if movie_id in movie_data:
                movie = movie_data[movie_id]
                # Check for null/NaN values
                title = movie['movie_title'] if pd.notna(
                    movie['movie_title']) else None
                year = int(movie['year_released']) if pd.notna(
                    movie['year_released']) else None

                # Extract genre information
                genres = []
                if pd.notna(movie['genres']) and movie['genres'] != '[]':
                    try:
                        # Parse genres as JSON
                        genres = json.loads(movie['genres'])
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse genres for movie {movie_id}")

                # Determine primary genre for frontend display
                primary_genre = genres[0] if genres else "No Genre"

                # Get poster URL if available
                poster_url = None
                if pd.notna(movie.get('image_url')):
                    base_url = "https://a.ltrbxd.com/resized"
                    image_path = f"{movie['image_url']}.jpg"
                    poster_url = f"{base_url}/{image_path}"

                recommendations.append({
                    'movie_id': str(movie_id),
                    'title': title,
                    'year': year,
                    'predicted_rating': rating,
                    'genres': genres,
                    'genre': primary_genre,  # Add primary genre for frontend
                    'poster_url': poster_url
                })
            else:
                logger.warning(f"Movie ID {movie_id} not found in movies data")

        return jsonify({
            'user_id': username,
            'recommendations': recommendations
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
