# Movie Sentiment Analysis

A machine learning project comparing traditional and deep learning approaches for sentiment classification on IMDB movie reviews.

## Overview

This project implements two different models to classify movie reviews as positive or negative:
- **Logistic Regression** with TF-IDF features
- **LSTM Neural Network** with word embeddings

## Dataset

- **Source**: IMDB Dataset of 50K Movie Reviews (Kaggle)
- **Size**: 26,790 reviews after cleaning
- **Balance**: 50.2% positive, 49.8% negative reviews
- **Format**: Text reviews with binary sentiment labels

## Models & Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Logistic Regression | 88.9% | 88.9% | 88.9% | 88.9% |
| LSTM | 86.4% | 86.5% | 86.5% | 86.5% |

## Key Features

- Comprehensive text preprocessing (HTML removal, tokenization, lemmatization)
- TF-IDF vectorization for traditional ML
- Word embeddings and sequence padding for deep learning
- Hyperparameter tuning with GridSearchCV
- Data visualization and exploratory analysis

## Technologies Used

- **Python Libraries**: pandas, numpy, scikit-learn, tensorflow/keras, nltk
- **Visualization**: matplotlib, seaborn, wordcloud
- **Text Processing**: BeautifulSoup, NLTK tokenizers

## Project Structure  
```bash
Sentiment_Analysis_Group12/
├── dataset/
│   └── IMDB Dataset.csv          # Raw dataset (50K movie reviews)
├── models/
│   ├── best_logistic_regression.joblib  # Trained Logistic Regression model
│   ├── best_lstm_model.keras            # Trained LSTM model (Keras)
│   ├── tfidf_vectorizer.joblib          # Saved TF-IDF vectorizer
│   ├── lstm_tokenizer.pickle            # Tokenizer for LSTM input
│   └── tokenizer.pickle                 # Backup tokenizer
├── notebooks/
│   ├── sentiment_analysis.ipynb         # EDA/preprocessing and All Trainings
└── README.md                            # This file
```
## Setup & Execution
#### **Clone the repository**
```
   git clone https://github.com/yourusername/Sentiment_Analysis_Group12.git
cd Sentiment_Analysis_Group12
```
#### **Quick Start**

1. Install required packages:
```bash
pip install pandas numpy scikit-learn tensorflow nltk matplotlib seaborn wordcloud beautifulsoup4
```

2. Download NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
3. Run the notebook or script to train both models

## Key Insights

- Balanced dataset eliminates class imbalance issues
- Positive reviews tend to be longer and more detailed
- Traditional ML (Logistic Regression) outperformed deep learning on this dataset
- Strong vocabulary patterns distinguish positive vs negative sentiment
