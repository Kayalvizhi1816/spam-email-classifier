import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Read the dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Original dataset information
print("Original dataset shape:")
print(data.shape)

print("\nDuplicate rows:")
print(data.duplicated().sum())

# Remove duplicate rows
data = data.drop_duplicates()

print("\nAfter removing duplicates:")
print(data.shape)

# Spam and Ham count
print("\nSpam and Ham count:")
print(data["label"].value_counts())

# Separate input and output
X = data["message"]
y = data["label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF testing shape:")
print(X_test_tfidf.shape)

# Create Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

print("\nModel training completed! 🎉")

# Make predictions on test data
predictions = model.predict(X_test_tfidf)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(predictions)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(accuracy * 100, "%")

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Test with a custom message
message = ["Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 now!"]

message_tfidf = vectorizer.transform(message)

result = model.predict(message_tfidf)

print("\nCustom Message:")
print(message[0])

print("\nPrediction:")
print(result[0])

# Save the trained model and TF-IDF vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully! 💾")