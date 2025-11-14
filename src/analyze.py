import numpy as np
import joblib

# Load model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def analyze_email_text(text, top_k=8):
    x_tfidf = vectorizer.transform([text])
    prob = model.predict_proba(x_tfidf)[0][1]
    label = "Phishing" if prob > 0.5 else "Legitimate"

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    tfidf_vals = x_tfidf.toarray()[0]
    present_idx = np.where(tfidf_vals > 0)[0]

    print("\n--- ANALYSIS RESULT ---")
    print(f"Prediction: {label}")
    print(f"Confidence (probability of phishing): {prob*100:.2f}%")

    if len(present_idx) == 0:
        print("(No words from this email matched the model vocabulary.)")
        return

    contributions = coefs[present_idx] * tfidf_vals[present_idx]
    top_pos_order = np.argsort(contributions)[-top_k:][::-1]
    top_pos = [(feature_names[present_idx[i]], contributions[present_idx[i]]) for i in top_pos_order]

    print("\nTop words contributing TO phishing:")
    for w, c in top_pos:
        print(f"  {w} ({c:.4f})")

if __name__ == "__main__":
    email = input("Paste your email here:\n")
    analyze_email_text(email)
