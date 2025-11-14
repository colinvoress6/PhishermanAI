# feature_pipeline.py
import re
import hashlib
from datetime import datetime
from email import message_from_string
from dateutil import parser as dateparser

import pandas as pd
import numpy as np
import tldextract

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import sparse

import lightgbm as lgb

# ---------- Text utilities ----------
URL_REGEX = re.compile(
    r'(https?://[^\s]+|www\.[^\s]+)', re.IGNORECASE
)

EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+', re.IGNORECASE)

URGENCY_WORDS = [
    'urgent', 'immediately', 'asap', 'verify', 'verification', 'password', 'suspend', 'billing', 'invoice',
    'pay now', 'action required', 'reset', 'locked'
]


def find_urls(text):
    if not isinstance(text, str):
        return []
    return URL_REGEX.findall(text)


def count_urls(text):
    return len(find_urls(text))


def extract_link_domains(text):
    urls = find_urls(text)
    domains = []
    for u in urls:
        # normalize
        if u.startswith('www.'):
            u = 'http://' + u
        try:
            ext = tldextract.extract(u)
            if ext.domain:
                domains.append(f"{ext.domain}.{ext.suffix}".lower())
        except Exception:
            continue
    return domains


def has_urgency(text):
    if not isinstance(text, str):
        return 0
    t = text.lower()
    return int(any(w in t for w in URGENCY_WORDS))


def is_html(text):
    if not isinstance(text, str):
        return 0
    # simple heuristic
    return int(bool(re.search(r'<(html|body|div|table|a|span|img)', text, re.IGNORECASE)))


# ---------- Header utilities ----------
def parse_from_field(from_field):
    """
    Returns a tuple (display_name, email_address)
    from_field examples:
      - "Alice Smith <alice@example.com>"
      - "alice@example.com"
      - '"Bank Support" <no-reply@bank.com>'
    """
    if not isinstance(from_field, str):
        return ("", "")
    # email address
    email_match = EMAIL_REGEX.search(from_field)
    email_addr = email_match.group(0).lower() if email_match else ""
    # display name is anything before the email
    display = from_field.split('<')[0].strip().strip('"') if '<' in from_field else ""
    if display == email_addr:
        display = ""
    return (display, email_addr)


def domain_from_email(email_addr):
    try:
        return email_addr.split('@')[-1].lower() if email_addr else ""
    except Exception:
        return ""


def display_name_mismatch(display, email_addr):
    # quick fuzzy mismatch: display contains a company name that is not the domain
    if not display or not email_addr:
        return 0
    domain = domain_from_email(email_addr)
    display_norm = re.sub(r'[^a-z0-9]', '', display.lower())
    domain_norm = re.sub(r'[^a-z0-9]', '', domain.split('.')[0] if '.' in domain else domain)
    return int(display_norm and (display_norm != domain_norm))


# ---------- Feature extraction per-row ----------
def extract_features_row(row):
    """
    Expects row with at least: 'subject', 'body', 'from', 'date', and optionally 'attachments' or 'has_attachments' fields.
    Returns dict of metadata features and a combined_text string for TF-IDF.
    """
    subject = row.get('subject', '') or ''
    body = row.get('body', '') or ''
    from_field = row.get('from', '') or row.get('sender', '')
    attachments_flag = 1 if row.get('has_attachments', False) or row.get('attachments') else 0
    date_val = row.get('date', None)

    display, email_addr = parse_from_field(from_field)
    sender_domain = domain_from_email(email_addr)
    # features
    urls_in_subject = find_urls(subject)
    urls_in_body = find_urls(body)

    link_domains = extract_link_domains(subject + '\n' + body)
    unique_link_domains = list(set(link_domains))

    combined_text = (subject + ' ') + (body[:5000])  # truncate body to avoid huge inputs

    feats = {
        'sender_domain': sender_domain,
        'display_name': display,
        'sender_email': email_addr,
        'display_mismatch': display_name_mismatch(display, email_addr),
        'num_urls': len(urls_in_subject) + len(urls_in_body),
        'num_unique_link_domains': len(unique_link_domains),
        'num_dots_in_from_domain': sender_domain.count('.'),
        'has_attachments': int(attachments_flag),
        'subject_len': len(subject),
        'body_len': len(body),
        'has_urgency': has_urgency(subject + ' ' + body),
        'is_html': is_html(body),
    }

    # optional: timestamp features
    try:
        if date_val:
            dt = dateparser.parse(date_val) if isinstance(date_val, str) else date_val
            feats['hour_of_day'] = dt.hour
            feats['weekday'] = dt.weekday()
        else:
            feats['hour_of_day'] = -1
            feats['weekday'] = -1
    except Exception:
        feats['hour_of_day'] = -1
        feats['weekday'] = -1

    return feats, combined_text


def build_feature_dataframe(df):
    """
    Input: pandas DataFrame with columns: subject, body, from, date, label (1/0)
    Output: (meta_df, text_series, labels)
    """
    meta_rows = []
    text_list = []
    for _, row in df.iterrows():
        feats, combined_text = extract_features_row(row)
        meta_rows.append(feats)
        text_list.append(combined_text)
    meta_df = pd.DataFrame(meta_rows)
    return meta_df, pd.Series(text_list), df['label'].reset_index(drop=True)


# ---------- Feature engineering for ML ----------
def prepare_feature_matrix(meta_df, text_series, tfidf_vectorizer=None):
    """
    Returns sparse matrix X and fitted tfidf_vectorizer
    - meta_df: DataFrame of metadata numeric/categorical features
    - text_series: Series of combined text for TF-IDF
    """
    # Basic numeric features
    numeric_cols = [
        'display_mismatch', 'num_urls', 'num_unique_link_domains', 'num_dots_in_from_domain',
        'has_attachments', 'subject_len', 'body_len', 'has_urgency', 'is_html',
        'hour_of_day', 'weekday'
    ]
    X_num = meta_df[numeric_cols].fillna(-1).astype(float).values

    # One-hot / target-free encoding for sender_domain (top-K)
    top_k = 100
    top_domains = meta_df['sender_domain'].value_counts().nlargest(top_k).index.tolist()
    domain_ohe = np.zeros((len(meta_df), len(top_domains)), dtype=float)
    domain_to_idx = {d: i for i, d in enumerate(top_domains)}
    for i, d in enumerate(meta_df['sender_domain'].fillna('')):
        if d in domain_to_idx:
            domain_ohe[i, domain_to_idx[d]] = 1.0

    # TF-IDF for text
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=25000,
            ngram_range=(1, 2),
            analyzer='word',
            min_df=3,
            max_df=0.9
        )
        X_text = tfidf_vectorizer.fit_transform(text_series.fillna(''))
    else:
        X_text = tfidf_vectorizer.transform(text_series.fillna(''))

    # combine
    X_meta = np.hstack([X_num, domain_ohe])
    X_meta_sparse = sparse.csr_matrix(X_meta)
    X = sparse.hstack([X_meta_sparse, X_text], format='csr')

    return X, tfidf_vectorizer


# ---------- Model training and evaluation ----------
def train_evaluate_lgb(X, y, params=None, n_splits=5, random_state=42):
    """
    Trains LightGBM with StratifiedKFold and prints per-fold metrics.
    Returns list of models per fold and aggregated metrics.
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_threads': 4,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'verbosity': -1,
            # use scale_pos_weight if labels imbalanced or pass class_weight in Dataset
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = []
    metrics = []
    fold = 0
    for train_idx, valid_idx in skf.split(X, y):
        fold += 1
        print(f"--- Fold {fold} ---")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y.iloc[train_idx].values, y.iloc[valid_idx].values

        # compute scale_pos_weight
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / (pos + 1e-9)
        params['scale_pos_weight'] = scale_pos_weight

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

        mdl = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        models.append(mdl)

        # predict
        y_pred_proba = mdl.predict(X_valid, num_iteration=mdl.best_iteration)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(y_valid, y_pred, average='binary', pos_label=1)
        ap = average_precision_score(y_valid, y_pred_proba)
        auc = roc_auc_score(y_valid, y_pred_proba)
        cm = confusion_matrix(y_valid, y_pred)

        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')
        print('Confusion matrix:\n', cm)
        metrics.append({
            'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'ap': ap, 'confusion_matrix': cm
        })
    # aggregate
    avg_metrics = {k: np.mean([m[k] for m in metrics if k != 'confusion_matrix']) for k in metrics[0].keys() if k != 'confusion_matrix'}
    print('--- Average metrics across folds ---')
    for k, v in avg_metrics.items():
        print(f'{k}: {v:.4f}')
    return models, metrics


# ---------- Example full-run function ----------
def run_full_pipeline(df):
    """
    df: pandas DataFrame with columns subject, body, from, date, label
    """
    meta_df, text_series, labels = build_feature_dataframe(df)
    X, tfidf = prepare_feature_matrix(meta_df, text_series, tfidf_vectorizer=None)
    models, metrics = train_evaluate_lgb(X, labels, n_splits=5)
    return {
        'models': models,
        'tfidf': tfidf,
        'metrics': metrics,
        'meta_df': meta_df
    }


# ---------- Utility: save feature CSV for inspection ----------
def save_features_for_inspection(meta_df, text_series, labels, path='features_preview.csv', nrows=5000):
    out = meta_df.copy()
    out['text_sample'] = text_series.fillna('').str[:1000]
    out['label'] = labels.values
    out.head(nrows).to_csv(path, index=False)
    print(f"Saved preview to {path}")
