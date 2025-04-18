import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import MovieCard from './components/MovieCard';
import GenreFilter from './components/GenreFilter';
import './App.css';

function App() {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedGenre, setSelectedGenre] = useState('All');
  const [uniqueGenres, setUniqueGenres] = useState([]);
  const [selectedPersona, setSelectedPersona] = useState(null);

  useEffect(() => {
    if (recommendations.length > 0) {
      // Extract all unique genres from recommendations
      const allGenres = new Set();
      recommendations.forEach(movie => {
        if (movie.genre) {
          allGenres.add(movie.genre);
        }
      });
      setUniqueGenres(Array.from(allGenres).sort());
    }
  }, [recommendations]);

  const handlePersonaSelect = async (persona) => {
    setSelectedPersona(persona);
    setLoading(true);
    setError(null);
    setSelectedGenre('All'); // Reset genre filter when fetching new recommendations

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          persona: persona
        }),
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

  // Filter recommendations based on selected genre
  const filteredRecommendations = selectedGenre === 'All'
    ? recommendations
    : recommendations.filter(movie => movie.genre === selectedGenre);

  // Group filtered movies by genre
  const groupedMovies = filteredRecommendations.reduce((acc, movie) => {
    const genre = movie.genre || 'No Genre';
    if (!acc[genre]) {
      acc[genre] = [];
    }
    acc[genre].push(movie);
    return acc;
  }, {});

  return (
    <div className="app">
      <Navbar onPersonaSelect={handlePersonaSelect} />

      <main className="main-content">
        {error && <div className="error-message">{error}</div>}

        {loading ? (
          <div className="loading-container">
            <div className="loading-text">Loading recommendations...</div>
          </div>
        ) : (
          <>
            {recommendations.length > 0 && (
              <div className="genre-filter">
                <GenreFilter
                  genres={uniqueGenres}
                  selectedGenre={selectedGenre}
                  onGenreSelect={setSelectedGenre}
                />
              </div>
            )}

            <div className="recommendations">
              {Object.entries(groupedMovies).map(([genre, movies]) => (
                <div key={genre} className="genre-section">
                  <h2 className="genre-title">{genre}</h2>
                  <div className="movie-grid">
                    {movies.map((movie) => (
                      <MovieCard key={movie.movie_id} movie={movie} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
