import joblib

# Load saved model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def test_email(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    confidence = model.predict_proba(text_vector).max()

    if prediction == 1:
        print(f"⚠️ This email looks like phishing! (confidence: {confidence:.2f})")
    else:
        print(f"✅ This email looks safe. (confidence: {confidence:.2f})")

# Interactive input
if __name__ == "__main__":
    print("Paste your email below (press ENTER twice to finish):\n")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    email_text = "\n".join(lines)
    test_email(email_text)
