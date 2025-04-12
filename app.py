from flask import Flask, request, jsonify
import torch
import os

app = Flask(__name__)

# Load the model
model_path = os.getenv('MODEL_PATH', 'model.pt')
model = torch.jit.load(model_path)
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_id = torch.tensor([data['user_id']])
        movie_id = torch.tensor([data['movie_id']])

        with torch.no_grad():
            prediction = model(user_id, movie_id)

        return jsonify({
            'rating': prediction.item(),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
