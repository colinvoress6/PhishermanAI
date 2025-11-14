# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    # Find all CSV files in the current folder
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the current folder. Place your data here.")
        return

    # Load and combine all CSVs
    df_list = []
    for file in csv_files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        df_list.append(df)
    data = pd.concat(df_list, ignore_index=True)
    print(f"\nLoaded {len(data)} emails from {len(csv_files)} files.")

    # Make sure 'text_combined' and 'label' columns exist
    if 'text_combined' not in data.columns or 'label' not in data.columns:
        print("Error: Each CSV must have 'text_combined' and 'label' columns.")
        return

    # Fill missing text with empty string
    data['text_combined'] = data['text_combined'].fillna('')

    # Features and labels
    X = data['text_combined']
    y = data['label']

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(model, "phishing_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("\n✅ Model and vectorizer saved successfully!")

if __name__ == "__main__":
    main()
