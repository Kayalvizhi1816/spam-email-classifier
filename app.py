from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("spam_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get message from HTML form
    message = request.form["message"]
    if not message.strip():
      return render_template(
        "index.html",
        error="Please enter a message!",
        message=""
    )

    # Convert message into TF-IDF
    message_tfidf = vectorizer.transform([message])

    # Predict
    result = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf).max() * 100

    # Send result back to HTML
    return render_template(
    "index.html",
    prediction=result,
    confidence=round(probability, 2),
    message=message
)

if __name__ == "__main__":
    app.run(debug=True)