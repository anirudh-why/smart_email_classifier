import os
import re

def extract_body_from_email(email_path):
    """
    Extracts the body from an email file (skipping headers).
    """
    with open(email_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    in_body = False
    body_lines = []
    for line in lines:
        # Headers and body are separated by a blank line
        if in_body:
            body_lines.append(line)
        elif line == '\n':
            in_body = True
    return ''.join(body_lines)

def clean_text(text):
    """
    Basic text cleaning: lowercase, remove non-letters, etc.
    """
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)           # remove HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text)        # keep only letters
    text = re.sub(r'\s+', ' ', text)             # collapse whitespace
    return text.strip()

def load_emails_from_dir(directory, label, max_files=None):
    """
    Loads all emails from a directory, extracts and cleans body, and assigns label.
    Returns list of (text, label) tuples.
    """
    emails = []
    files = os.listdir(directory)
    if max_files:
        files = files[:max_files]
    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            body = extract_body_from_email(filepath)
            cleaned = clean_text(body)
            if cleaned:  # only add non-empty emails
                emails.append((cleaned, label))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return emails