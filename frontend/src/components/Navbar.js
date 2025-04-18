import React from 'react';
import './Navbar.css';
import NotificationBar from './NotificationBar';

const Navbar = ({ onPersonaSelect }) => {
    return (
        <div className="navbar-container">
            <nav className="navbar">
                <div className="navbar-left">
                    <span className="navbar-brand">🎬 Movie Match</span>
                </div>
                <div className="navbar-right">
                    <a href="#watchlist" className="nav-link">Watchlist</a>
                    <a href="#recommendations" className="nav-link">Recommendations</a>
                </div>
            </nav>
            <NotificationBar onPersonaSelect={onPersonaSelect} />
        </div>
    );
};

export default Navbar; 