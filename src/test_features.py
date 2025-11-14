# test_features.py
import pandas as pd
from email_features import build_feature_dataframe

def main():
    # Load dataset
    try:
        df = pd.read_csv("data/phishing_email.csv")
    except FileNotFoundError:
        print("Error: data/phishing_email.csv not found. Make sure your dataset is in the data/ folder.")
        return

    # Extract features
    features = build_feature_dataframe(df)

    # Show basic info
    print("Feature extraction successful!")
    print(f"Number of emails: {len(df)}")
    print(f"Number of features: {features.shape[1]}")
    
    # Show the first 5 rows
    print("\nFirst 5 rows of the feature dataframe:")
    print(features.head())

if __name__ == "__main__":
    main()
