from flask import Flask, request, jsonify, send_from_directory
import torch
import os
import pandas as pd
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)

# Load model and mappings
model = torch.load('model.pt', map_location=torch.device('cpu'))
model.eval()

# Load ratings data for watched movies
ratings_df = pd.read_csv('data/processed/ratings.csv')


@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_id = data.get('user_id')

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Get list of movies user has already watched
        watched_movies = set(
            ratings_df[ratings_df['user_id'] == user_id]['movie_id'].tolist())

        # Create tensors for all movies
        movie_ids = torch.arange(model.num_movies)
        user_ids = torch.full((model.num_movies,), int(user_id))

        # Get predictions for all movies
        with torch.no_grad():
            predictions = model(user_ids, movie_ids)

        # Create a mask for unwatched movies
        unwatched_mask = ~torch.tensor(
            [movie_id in watched_movies for movie_id in range(model.num_movies)])

        # Apply mask to predictions
        unwatched_predictions = predictions[unwatched_mask]
        unwatched_movie_ids = movie_ids[unwatched_mask]

        # Get top 10 predictions
        top_k = 10
        top_values, top_indices = torch.topk(unwatched_predictions, k=top_k)

        # Convert to list format
        recommendations = []
        for value, idx in zip(top_values, top_indices):
            movie_id = unwatched_movie_ids[idx].item()
            recommendations.append({
                'movie_id': str(movie_id),
                'predicted_rating': value.item()
            })

        return jsonify({
            'recommendations': recommendations,
            'total_unwatched': len(unwatched_movie_ids)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
