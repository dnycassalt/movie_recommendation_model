import React, { useState } from 'react';
import './NotificationBar.css';

const NotificationBar = ({ onPersonaSelect }) => {
    const [selectedPersona, setSelectedPersona] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (selectedPersona) {
            onPersonaSelect(selectedPersona);
        }
    };

    return (
        <div className="notification-bar">
            <div className="notification-content">
                <p className="notification-text">
                    Welcome to Movie Match!
                    This prototype uses a collaborative filtering model that leverages <a href="https://www.kaggle.com/datasets/samlearner/letterboxd-movie-ratings-data">user rating and movie behavior from Letterboxd</a> to provide recommendations.
                    Please select a randomized persona from the dropdown to see the recommendations in action.
                </p>
                <select
                    className="persona-select"
                    value={selectedPersona}
                    onChange={(e) => setSelectedPersona(e.target.value)}
                >
                    <option value="">Choose a persona...</option>
                    <option value="persona_1">Persona 1</option>
                    <option value="persona_2">Persona 2</option>
                    <option value="persona_3">Persona 3</option>
                    <option value="persona_4">Persona 4</option>
                    <option value="persona_5">Persona 5</option>
                </select>
                <button
                    className="submit-button"
                    onClick={handleSubmit}
                    disabled={!selectedPersona}
                >
                    Get Recommendations
                </button>
            </div>
        </div>
    );
};

export default NotificationBar; 