# Smart Email Classifier

## Description

A modular Python project for classifying emails (such as spam vs. ham) using machine learning and natural language processing. This project demonstrates a clean and maintainable pipeline—from raw email extraction and text preprocessing to feature engineering and final prediction, including confidence scores for each classification.

## Features

- End-to-end pipeline: raw email → cleaned text → features → prediction
- Modular and reusable functions for each step
- Handles class imbalance in training data
- Extracts both TF-IDF text features and custom features (URLs, exclamation marks, capitalization ratio, email length, etc.)
- Outputs prediction labels along with confidence scores
- Example script provided for easy testing
