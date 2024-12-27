# Movie Recommendation System

## Overview
This project implements a movie recommendation system using machine learning techniques. The system analyzes a dataset of movies and their associated metadata to recommend similar movies based on user input. The recommendations are generated using natural language processing (NLP) techniques, including text vectorization and cosine similarity.

## Dataset
The project uses two datasets:
1. **tmdb_5000_movies.csv**: Contains information about movies, including titles, overviews, genres, and keywords.
2. **tmdb_5000_credits.csv**: Contains information about the cast and crew of the movies.

## Features
- Extracts relevant information from the datasets, including genres, keywords, cast, and crew.
- Preprocesses the data by normalizing text, removing spaces, and stemming words.
- Creates a bag-of-words model to convert text data into numerical vectors.
- Calculates cosine similarity between movies to find similar titles.
- Provides a function to recommend movies based on user input.

## Requirements
To run this project, you will need the following Python libraries:
- pandas
- numpy
- nltk
- scikit-learn

You can install the required libraries using pip:

```bash
pip install pandas numpy nltk scikit-learn
