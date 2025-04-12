# Movie Recommendation API

## Overview
This is a Flask-based API that serves a React frontend and provides movie recommendations using a PyTorch model.

## Directory Structure
```
movie_recommendation/
├── api/                 # Flask backend
│   ├── app.py          # Main Flask application
│   └── README.md       # This file
├── db/                 # Database configuration
│   ├── Makefile        # Database setup and management
│   └── schema.sql      # Database schema
├── frontend/           # React frontend
│   ├── public/         # Static files
│   ├── src/            # React source code
│   │   ├── App.js      # Main React component
│   │   └── App.css     # Styles
│   └── build/          # Production build
└── data/               # Data files
```

## Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn
- PyTorch
- Flask
- Flask-CORS

## Setup Instructions

### Backend Setup
1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the project:
```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
```

### Frontend Setup
1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Build the frontend:
```bash
npm run build
```

### Database Setup
1. Navigate to the db directory:
```bash
cd db
```

2. Create the database schema:
```bash
make create-schema
```

3. Import the data:
```bash
make import-data
```

## Running the Application

### Development Mode
1. Start the Flask backend:
```bash
python api/app.py
```

2. In a separate terminal, start the React development server:
```bash
cd frontend
npm start
```

### Production Mode
1. Build the React frontend:
```bash
cd frontend
npm run build
```

2. Start the Flask server (it will serve the built frontend):
```bash
python api/app.py
```

The application will be available at `http://localhost:5000`.

## API Endpoints

### Frontend Routes
- `/` - Serves the React frontend
- `/static/*` - Serves static assets

### API Endpoints
- `GET /api/health` - Health check endpoint
- `POST /api/predict` - Get movie recommendations
  - Request body: `{"user_id": "string"}`
  - Response: 
    ```json
    {
      "recommendations": [
        {
          "movie_id": "string",
          "predicted_rating": number
        }
      ],
      "total_unwatched": number
    }
    ```

## Frontend Features
- User input form for entering user ID
- Real-time loading states
- Error handling and display
- Responsive design
- Movie recommendations display with ratings

## Deployment
The application can be deployed to Google Cloud Run. See the deployment documentation for details.

## Troubleshooting
- If the frontend isn't loading, ensure you've built it with `npm run build`
- If API calls fail, check the browser console for errors
- Ensure CORS is properly configured if running frontend and backend separately
- For database issues, check the db directory for setup instructions

## Support
For additional help, please contact the development team. 