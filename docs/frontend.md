# Frontend Documentation

## Overview
The movie recommendation system includes a React-based frontend that provides a user-friendly interface for interacting with the recommendation API. The frontend is built using modern React practices and is designed to be responsive and user-friendly.

## Architecture

### Components
- `App.js`: Main application component
  - Handles user input
  - Manages API communication
  - Displays recommendations
  - Handles loading and error states

### Styling
- `App.css`: Custom styles for the application
  - Responsive design
  - Modern UI elements
  - Loading states
  - Error displays

## Development

### Setup
1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

### Building for Production
```bash
npm run build
```

The build output will be in the `frontend/build` directory, which is served by the Flask backend.

## Features

### User Interface
- Clean, modern design
- Responsive layout
- Loading indicators
- Error messages
- Movie recommendations display

### Functionality
- User ID input
- Real-time API communication
- Error handling
- Loading states
- Recommendation display

## API Integration

### Endpoints
- `POST /api/predict`
  - Request: `{"user_id": "string"}`
  - Response: List of movie recommendations with predicted ratings

### Error Handling
- Network errors
- API errors
- Invalid input
- Loading states

## Best Practices

### Code Organization
- Component-based architecture
- Separation of concerns
- Reusable components
- Clean code practices

### Performance
- Lazy loading
- Optimized builds
- Efficient state management
- Minimal re-renders

### Security
- Input validation
- CORS configuration
- Secure API communication
- Error handling

## Deployment

### Production Build
1. Build the frontend:
```bash
npm run build
```

2. The Flask backend will serve the built files from `frontend/build`

### Development
- Use `npm start` for development
- Hot reloading enabled
- Development server on port 3000
- Proxy to backend API

## Troubleshooting

### Common Issues
1. Frontend not loading
   - Check if build was successful
   - Verify Flask is serving static files
   - Check browser console for errors

2. API communication issues
   - Verify CORS configuration
   - Check network requests
   - Validate API responses

3. Styling issues
   - Check CSS imports
   - Verify responsive design
   - Test on different browsers

## Support
For frontend-related issues, please contact the development team or open an issue in the project repository. 