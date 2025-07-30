import os

# Paths to ham and spam
ham_dir = "data/raw/easy_ham"
spam_dir = "data/raw/spam"

# List a few ham emails
ham_files = os.listdir(ham_dir)[:3]
print("Sample Ham Emails:")
for fname in ham_files:
    with open(os.path.join(ham_dir, fname), 'r', encoding='latin-1') as f:
        print(f"\n--- {fname} ---")
        print(f.read()[:500])  # Print first 500 chars

# List a few spam emails
spam_files = os.listdir(spam_dir)[:3]
print("\nSample Spam Emails:")
for fname in spam_files:
    with open(os.path.join(spam_dir, fname), 'r', encoding='latin-1') as f:
        print(f"\n--- {fname} ---")
        print(f.read()[:500])