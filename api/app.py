import os
import torch
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)

# Number of movies in the dataset
NUM_MOVIES = 286071

# Load user mapping
users_path = os.path.join('..', 'data', 'processed', 'users.csv')
try:
    users_df = pd.read_csv(users_path)
    # Create mapping from username to numeric index
    username_to_id = {username: idx for idx,
                      username in enumerate(users_df['username'])}
    logger.info(f"Loaded {len(username_to_id)} users from {users_path}")
except Exception as e:
    logger.error(f"Error loading users data: {str(e)}")
    raise

# Load the model
model_path = os.getenv('MODEL_PATH', 'model_prepared.pt')
logger.info(f"Loading model from {model_path}")

try:
    model = torch.load(model_path, weights_only=False)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise


@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        username = data.get('user_id')  # Actually a username

        if not username:
            return jsonify({"error": "Missing user_id"}), 400

        # Convert username to numeric ID
        if username not in username_to_id:
            return jsonify({"error": f"User {username} not found"}), 404

        user_id = username_to_id[username]
        user_tensor = torch.tensor([user_id], dtype=torch.long)

        # Get predictions for all movies
        predictions = []
        for movie_id in range(NUM_MOVIES):
            movie_tensor = torch.tensor([movie_id], dtype=torch.long)
            with torch.no_grad():
                prediction = model(user_tensor, movie_tensor)
                # Clamp predictions between 0 and 10
                prediction = torch.clamp(prediction, 0, 10)
                predictions.append((movie_id, prediction.item()))

        # Sort predictions and get top 50
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_50 = predictions[:50]

        # Convert to list format
        recommendations = [
            {
                'movie_id': str(movie_id),
                'predicted_rating': rating
            }
            for movie_id, rating in top_50
        ]

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
