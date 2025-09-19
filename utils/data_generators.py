"""
Data generation utilities for ML fundamentals course.

Creates synthetic datasets that extend the "Go Dolphins!" theme
while maintaining educational value.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import re

def create_sports_tweets_features(text: str) -> Dict[str, float]:
    """
    Extract features from a sports tweet for sentiment analysis.

    This is the feature engineering function students will modify and extend.

    Args:
        text: Raw tweet text

    Returns:
        Dictionary of feature name -> value pairs
    """
    text_lower = text.lower()

    # Basic features from the articles
    features = {
        'word_count': len(text.split()),
        'has_team': 1.0 if any(team in text_lower for team in
                              ['dolphins', 'fins', 'miami']) else 0.0,
        'has_exclamation': 1.0 if '!' in text else 0.0,
    }

    return features

def extract_features_batch(texts: List[str], feature_function=create_sports_tweets_features) -> np.ndarray:
    """
    Apply feature extraction to a batch of texts.

    Args:
        texts: List of text strings
        feature_function: Function to extract features from single text

    Returns:
        (n_samples, n_features) numpy array
    """
    feature_dicts = [feature_function(text) for text in texts]

    # Get feature names consistently
    feature_names = list(feature_dicts[0].keys())

    # Convert to numpy array
    features = np.array([[d[name] for name in feature_names] for d in feature_dicts])

    return features, feature_names

def load_sports_dataset(filepath: str = None) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load and preprocess the sports tweets dataset.

    Args:
        filepath: Path to CSV file (if None, uses relative path)

    Returns:
        features: (n_samples, n_features) array
        labels: (n_samples,) binary labels
        feature_names: List of feature names
        texts: Original text strings
    """
    if filepath is None:
        filepath = './Part1-Problems/data/sports_tweets.csv'

    df = pd.read_csv(filepath)

    # Extract features
    features, feature_names = extract_features_batch(df['text'].tolist())

    # Get labels
    labels = df['label'].values

    return features, labels, feature_names, df['text'].tolist()

def generate_synthetic_sports_data(n_samples: int = 100,
                                 noise_level: float = 0.1,
                                 random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 2D dataset for visualization exercises.

    Creates data that follows sentiment patterns:
    - Team mentions + excitement -> positive
    - Negative words -> negative
    - Add controlled noise for realism

    Args:
        n_samples: Number of samples to generate
        noise_level: Amount of random noise to add
        random_state: Random seed for reproducibility

    Returns:
        features: (n_samples, 2) feature matrix
        labels: (n_samples,) binary labels
    """
    np.random.seed(random_state)

    # Generate two main clusters
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # Positive sentiment cluster (high team mention + excitement)
    pos_features = np.random.multivariate_normal(
        mean=[0.8, 0.7],  # High values for both features
        cov=[[0.1, 0.05], [0.05, 0.1]],
        size=n_pos
    )
    pos_labels = np.ones(n_pos)

    # Negative sentiment cluster (low values)
    neg_features = np.random.multivariate_normal(
        mean=[0.2, 0.3],  # Low values
        cov=[[0.1, -0.02], [-0.02, 0.1]],
        size=n_neg
    )
    neg_labels = np.zeros(n_neg)

    # Combine and shuffle
    features = np.vstack([pos_features, neg_features])
    labels = np.hstack([pos_labels, neg_labels])

    # Add noise
    features += np.random.normal(0, noise_level, features.shape)

    # Clip to reasonable ranges
    features = np.clip(features, 0, 1)

    # Shuffle
    indices = np.random.permutation(n_samples)
    features = features[indices]
    labels = labels[indices]

    return features, labels

def create_expanded_sports_dataset(base_tweets: List[str],
                                 base_labels: List[int],
                                 expansion_factor: int = 5) -> Tuple[List[str], List[int]]:
    """
    Create expanded dataset with variations of base tweets.

    This generates more training data while maintaining the theme.

    Args:
        base_tweets: Original tweet texts
        base_labels: Original sentiment labels
        expansion_factor: How many variations to create per base tweet

    Returns:
        expanded_tweets: Larger list of tweet texts
        expanded_labels: Corresponding labels
    """
    # Templates for positive variations
    positive_templates = [
        "Go {}!",
        "Love the {}!",
        "{} are amazing!",
        "Great win by {}!",
        "{} played fantastically!",
        "What a game by {}!",
        "Incredible performance {}!",
        "{} are the best!",
        "Amazing season for {}!",
        "Perfect game {}!"
    ]

    # Templates for negative variations
    negative_templates = [
        "Terrible game {}",
        "Disappointed in {}",
        "{} played poorly",
        "Bad performance by {}",
        "Awful season for {}",
        "Can't stand {}",
        "{} are the worst",
        "Horrible game {}",
        "Frustrated with {}",
        "Terrible coaching {}"
    ]

    team_names = ["Dolphins", "the Fins", "Miami", "the team"]

    expanded_tweets = []
    expanded_labels = []

    for tweet, label in zip(base_tweets, base_labels):
        # Include original
        expanded_tweets.append(tweet)
        expanded_labels.append(label)

        # Generate variations
        templates = positive_templates if label == 1 else negative_templates

        for i in range(expansion_factor - 1):
            template = templates[i % len(templates)]
            team = team_names[i % len(team_names)]

            new_tweet = template.format(team)
            expanded_tweets.append(new_tweet)
            expanded_labels.append(label)

    return expanded_tweets, expanded_labels

def create_feature_engineering_challenges() -> List[Dict]:
    """
    Generate a set of feature engineering challenges for students.

    Returns list of challenge dictionaries with text and expected insights.
    """
    challenges = [
        {
            "text": "Love the fins!",
            "challenge": "How do you handle team nicknames vs official names?",
            "insight": "Need fuzzy matching or synonym detection"
        },
        {
            "text": "AMAZING GAME!!!!!",
            "challenge": "How do you handle multiple exclamation marks?",
            "insight": "Count total exclamations or cap at binary feature"
        },
        {
            "text": "Not bad at all",
            "challenge": "How do you handle negation?",
            "insight": "Simple word presence fails - need context understanding"
        },
        {
            "text": "😍🏈 Dolphins! 🐬",
            "challenge": "How do you handle emojis?",
            "insight": "Emojis carry strong sentiment signals"
        },
        {
            "text": "Dolphins win 28-7!",
            "challenge": "How do you extract score information?",
            "insight": "Regex patterns for scores, margin calculation"
        }
    ]

    return challenges