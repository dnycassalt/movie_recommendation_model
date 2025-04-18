import React from 'react';
import './GenreFilter.css';

const GenreFilter = ({ genres, selectedGenre, onGenreSelect }) => {
    return (
        <div className="genre-filter">
            <label htmlFor="genre-select">Filter by Genre:</label>
            <select
                id="genre-select"
                value={selectedGenre}
                onChange={(e) => onGenreSelect(e.target.value)}
                className="genre-select"
            >
                <option value="All">All Genres</option>
                {genres.map((genre, index) => (
                    <option key={index} value={genre}>
                        {genre}
                    </option>
                ))}
            </select>
        </div>
    );
};

export default GenreFilter; 