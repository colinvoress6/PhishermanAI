import pandas as pd
import re
import tldextract
from dateutil import parser

def build_feature_dataframe(df):
    """
    Extract features from a DataFrame of emails.
    Columns expected (if present): sender_email, display_name, subject, body, date, label
    Returns:
        meta_df: DataFrame of extracted features
        text_series: Series of email bodies
        labels: Series of labels (or None if missing)
    """

    def safe_column(col_name, default=''):
        return df[col_name].fillna(default) if col_name in df.columns else pd.Series([default]*len(df))

    meta_df = pd.DataFrame()

    # Sender info
    meta_df['sender_email'] = safe_column('sender_email', '')
    meta_df['sender_domain'] = meta_df['sender_email'].apply(lambda x: tldextract.extract(str(x)).domain if x else 'unknown')
    meta_df['display_name'] = safe_column('display_name', '')

    # Length features
    meta_df['subject_length'] = safe_column('subject', '').astype(str).apply(len)
    meta_df['body_length'] = safe_column('body', '').astype(str).apply(len)

    # HTML flag
    meta_df['is_html'] = safe_column('body', '').astype(str).apply(lambda x: int('<html>' in x.lower()))

    # Date features
    def parse_datetime(x):
        try:
            dt = parser.parse(str(x))
            return dt.hour, dt.weekday()
        except:
            return -1, -1

    hours, weekdays = zip(*safe_column('date', '').apply(parse_datetime))
    meta_df['hour_of_day'] = hours
    meta_df['weekday'] = weekdays

    # Link analysis
    meta_df['num_urls'] = safe_column('body', '').astype(str).apply(lambda x: len(re.findall(r'http[s]?://\S+', x)))

    def has_suspicious_link(row):
        sender_email = row.get('sender_email', '')
        sender_domain = tldextract.extract(str(sender_email)).domain if sender_email else ''
        body_text = str(row.get('body', ''))
        urls = re.findall(r'http[s]?://\S+', body_text)
        for url in urls:
            url_domain = tldextract.extract(url).domain
            if url_domain and url_domain != sender_domain:
                return 1
        return 0

    meta_df['suspicious_link'] = df.apply(lambda row: has_suspicious_link(row), axis=1)

    # Text for NLP
    text_series = safe_column('body', '')

    # Labels if available
    labels = safe_column('label', None)
    if labels.isnull().all():
        labels = None

    return meta_df, text_series, labels
