import os
import pandas as pd
from email_features import build_feature_dataframe

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/phishing_email.csv")

def main():
    df = pd.read_csv(DATA_PATH)
    features, text, labels = build_feature_dataframe(df)
    print("Analysis complete. Feature stats:")
    print(features.describe())

if __name__ == "__main__":
    main()
