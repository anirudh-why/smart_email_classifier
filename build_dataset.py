from data_preprocessing import load_emails_from_dir
import pandas as pd

ham_dir = "data/raw/easy_ham"
spam_dir = "data/raw/spam"

ham_emails = load_emails_from_dir(ham_dir, "ham")
spam_emails = load_emails_from_dir(spam_dir, "spam")

all_emails = ham_emails + spam_emails

# Create DataFrame
df = pd.DataFrame(all_emails, columns=['text', 'label'])

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV for later use
df.to_csv("data/email_dataset.csv", index=False)
print(f"Dataset built with {len(df)} emails. Saved to data/email_dataset.csv")