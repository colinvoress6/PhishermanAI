import pandas as pd
from email_features import build_feature_dataframe

def main():
    try:
        df = pd.read_csv("data/phishing_email.csv")
    except FileNotFoundError:
        print("Error: data/phishing_email.csv not found. Make sure your dataset is in the data/ folder.")
        return

    features, _, labels = build_feature_dataframe(df)

    print("Feature extraction successful!")
    print(f"Number of emails: {len(df)}")
    print(f"Number of features: {features.shape[1]}")
    print("\nFirst 5 rows of the feature dataframe:")
    print(features.head())

if __name__ == "__main__":
    main()
