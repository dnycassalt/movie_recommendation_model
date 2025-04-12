# Data Visualization Overview

## Introduction
The movie recommendation system includes several visualization tools to help understand and analyze the data. These tools are located in the `data-viz` directory and provide insights into movies, users, and ratings.

## Available Visualizations

### Movie Visualizations (`visualize_movies.py`)
- Movie release year distribution
- Genre distribution
- Popularity analysis
- Rating distribution
- Runtime analysis

### User Visualizations (`visualize_users.py`)
- User activity distribution
- Rating frequency
- User demographics
- User engagement metrics
- Rating patterns

### Rating Visualizations (`visualize_ratings.py`)
- Rating distribution
- Rating trends over time
- Rating patterns by genre
- User rating behavior
- Movie rating patterns

## Usage

### Prerequisites
- Python 3.8+
- Required packages:
  - pandas
  - matplotlib
  - seaborn
  - numpy

### Running Visualizations
1. Navigate to the data-viz directory:
```bash
cd data-viz
```

2. Run any visualization script:
```bash
python visualize_movies.py
python visualize_users.py
python visualize_ratings.py
```

### Output
- Each script generates visualizations in the `plots` directory
- Visualizations are saved as PNG files
- Some scripts may also output summary statistics to the console

## Customization
Each visualization script can be customized by:
- Modifying plot styles
- Adjusting figure sizes
- Changing color schemes
- Adding or removing specific visualizations

## Best Practices
- Always check data quality before visualization
- Use appropriate plot types for different data types
- Include clear labels and titles
- Save high-resolution images for presentations
- Document any data preprocessing steps

## Troubleshooting
Common issues and solutions:
1. Missing data
   - Check data file paths
   - Verify data format
   - Handle missing values appropriately

2. Plotting errors
   - Check matplotlib version
   - Verify data types
   - Ensure sufficient memory

3. Style issues
   - Check seaborn version
   - Verify style settings
   - Adjust figure sizes if needed

## Support
For visualization-related issues, please contact the development team or open an issue in the project repository. 