# IMDB Sentiment Analysis

This project performs sentiment analysis on IMDB movie reviews using various text processing techniques and machine learning algorithms.

## Purpose

The purpose of this project is to analyze the sentiment of IMDB movie reviews and build machine learning models to classify the reviews as positive or negative based on their sentiment.

## Instructions

1. Clone the repository to your local machine.
2. Install the required libraries using the `requirements.txt` file.
3. Download the IMDB dataset and place it in the project directory.
4. Run the provided Python script `sentiment_analysis.py`.

## Libraries Used

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computing.
- `matplotlib` and `seaborn`: For data visualization.
- `nltk`: For natural language processing tasks such as stemming, lemmatization, and stopword removal.
- `sklearn`: For machine learning algorithms and evaluation metrics.
- `gensim`: For training word embeddings using Word2Vec.
- `tqdm`: For displaying progress bars during data processing.

## Functions Used

### Data Preprocessing

1. **Text Cleaning**: Removes special characters and converts text to lowercase.
2. **Tokenization**: Splits text into individual words.
3. **Stemming**: Reduces words to their root form using Porter Stemmer.
4. **Stopword Removal**: Removes common English stopwords.

### Feature Extraction

1. **Bag of Words (BoW)**: Represents text data as a matrix of word occurrences.
2. **TF-IDF Vectorizer**: Computes TF-IDF (Term Frequency-Inverse Document Frequency) features from the text data.
3. **Word2Vec**: Generates word embeddings from the text corpus.
4. **Average Word2Vec**: Computes the average vector representation of words in each document.

### Machine Learning Models

1. **Multinomial Naive Bayes Classifier**: Used for BoW and TF-IDF features.
2. **Random Forest Classifier**: Utilized for Word2Vec embeddings.

## Results

- **BoW and TF-IDF**: Achieved an accuracy of [insert accuracy here] using Multinomial Naive Bayes Classifier.
- **Word2Vec**: Achieved an accuracy of [insert accuracy here] using Random Forest Classifier.
- **Average Word2Vec**: Achieved an accuracy of [insert accuracy here] using Random Forest Classifier.

## References
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data
