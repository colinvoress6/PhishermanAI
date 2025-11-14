import pickle
import pandas as pd
from email_features import build_feature_dataframe

def read_email(filename='email.txt'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            body = f.read()
        return {'sender_email': 'unknown@unknown.com', 'display_name': 'Unknown', 'subject': '', 'body': body, 'date': ''}
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} not found. Please create it in the project folder.")

def load_model():
    with open('models/phishing_model.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    email_dict = read_email()
    df = pd.DataFrame([email_dict])
    features, _, _ = build_feature_dataframe(df)

    model = load_model()
    prediction = model.predict(features)[0]

    print("=== Features Extracted ===")
    print(features)
    print(f"\nPrediction: {'Phishing' if prediction == 1 else 'Not Phishing'}")

if __name__ == "__main__":
    main()
