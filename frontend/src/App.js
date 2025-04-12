import React, { useState } from 'react';
import './App.css';

function App() {
  const [userId, setUserId] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_id: userId }),
      });

      if (!response.ok) {
        throw new Error('Failed to get recommendations');
      }

      const data = await response.json();
      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Movie Recommendations</h1>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Enter username"
            required
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Loading...' : 'Get Recommendations'}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {recommendations.length > 0 && (
          <div className="recommendations">
            <h2>Top Recommendations</h2>
            <div className="movie-grid">
              {recommendations.map((rec, index) => (
                <div key={index} className="movie-card">
                  <h3>{rec.title || 'Unknown Title'}</h3>
                  <p className="movie-year">{rec.year ? `(${rec.year})` : ''}</p>
                  <p className="movie-rating">Rating: {rec.predicted_rating.toFixed(1)}/10</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
