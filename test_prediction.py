from prediction import predict_email

spam_email = """
Congratulations! You have won a $1,000 gift card. Click here to claim your prize now: http://spammylink.com
"""

ham_email = """
Hi team,

The meeting is rescheduled to 3pm. Please confirm your availability.

Regards,
Manager Raman
"""

label1, confidence1 = predict_email(spam_email)
print(f"Spam test - Predicted label: {label1}, Confidence: {confidence1:.2f}")

label2, confidence2 = predict_email(ham_email)
print(f"Ham test - Predicted label: {label2}, Confidence: {confidence2:.2f}")