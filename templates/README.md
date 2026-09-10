# 📧 Spam Email Classifier

An AI/ML based web application that classifies SMS messages as **Spam** or **Ham (Safe)** using Natural Language Processing and Machine Learning.

## 🚀 Features

- 📩 Enter any SMS/message
- 🤖 AI-based spam detection
- 🔴 Detects spam messages
- 🟢 Detects safe messages
- 📊 Shows prediction confidence
- 🗑️ Clear message option
- ⚠️ Handles empty messages
- 🌐 Flask-based web interface

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Flask
- HTML
- CSS
- JavaScript

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset**.

The dataset contains SMS messages labelled as:

- `ham` — Normal/Safe message
- `spam` — Spam message

## 🧠 Machine Learning Model

The following steps are used:

1. Load the SMS dataset
2. Remove duplicate messages
3. Separate messages and labels
4. Split the dataset into training and testing data
5. Convert text into numerical features using TF-IDF
6. Train a Multinomial Naive Bayes classifier
7. Evaluate the model
8. Save the trained model
9. Use the model through a Flask web application

## 📈 Model Performance

The model achieved approximately **95% accuracy** on the test dataset.

## 📁 Project Structure

```text
Spam Email Classifier
│
├── dataset
│   └── SMSSpamCollection
│
├── templates
│   └── index.html
│
├── app.py
├── train_model.py
├── spam_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md