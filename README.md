# Spam Email Detection Web Application:-

    This project is a machine learning–based spam email detection system with a Flask web interface.
    It allows users to enter email or message text and instantly check whether it is Spam or Not Spam.

# Project Overview:-

    The system uses Natural Language Processing (NLP) techniques and Machine Learning models to    classify messages.
  
    Two different models are implemented and tested:

        Naive Bayes Classifier
        Logistic Regression

# The trained models are integrated into a Flask web app for real-time prediction.

    Project Structure
    ├── app.py                     # Flask app using Naive Bayes model
    ├── app_1.py                   # Flask app using Logistic Regression model
    ├── Naive_bayes_modal.py       # Training script for Naive Bayes
    ├── Logistic_reg_modal.py      # Training script for Logistic Regression
    ├── Naive_bayes_model.pkl      # Saved Naive Bayes model
    ├── logistic_reg_model.pkl     # Saved Logistic Regression model
    ├── word_dict.pkl              # Word dictionary for Naive Bayes
    ├── word_dict_1.pkl            # Word dictionary for Logistic Regression
    ├── style.css                  # Frontend styling
    ├── spam_ham_dataset.csv       # Dataset
    ├── logo.png                   # Project logo
    ├── templates/
    │   └── index.html             # Web interface
    └── README.md

# Dataset Description:-

    Dataset name: spam_ham_dataset.csv
  
    Type: Text classification dataset
  
    Classes:

        spam → Unwanted or fraudulent messages
        ham → Legitimate messages

    Columns:
    
        Column	Description
        label	Message category (spam / ham)
        text	Message content
    
# How It Works:-

    Dataset is loaded and cleaned
  
    Top 3000 most frequent words are selected
  
    Messages are converted into numerical feature vectors
  
    Models are trained using:
    
    Multinomial Naive Bayes
    
    Logistic Regression
    
    Trained models are saved using pickle
  
    Flask app loads the model and predicts user input in real time

# Web Application Features:-

    Clean and responsive UI
    Text input for message/email
    Instant spam prediction
    Visual result indicators
  
# Technologies Used:-

    Python
    Flask
    Pandas
    NumPy
    Scikit-learn
    HTML / CSS
    Pickle

# Use Cases

    Email spam filtering
    SMS spam detection
    NLP and ML academic projects
    Flask + ML integration practice
