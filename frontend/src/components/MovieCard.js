import React from 'react';
import './MovieCard.css';

const MovieCard = ({ movie }) => {
    if (!movie) return null;

    const fallbackImage = 'https://via.placeholder.com/300x450?text=No+Poster+Available';

    return (
        <div className={`movie-card ${movie.genre?.toLowerCase() || ''}`}>
            <div className="movie-image" style={{ backgroundImage: `url(${movie.poster_url || fallbackImage})` }}>
                <div className="movie-overlay">
                    <button className="watched-button">Watched</button>
                </div>
            </div>
            <div className="movie-info">
                <h3>{movie.title || 'Unknown Title'}</h3>
                <p className="genre">Genre: {movie.genre || 'Unknown Genre'}</p>
            </div>
        </div>
    );
};

export default MovieCard; 