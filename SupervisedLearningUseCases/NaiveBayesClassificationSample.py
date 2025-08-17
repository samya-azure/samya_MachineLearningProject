
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Sample data
emails = [
    "Win a free iPhone now",        # Spam
    "Limited offer just for you",   # Spam
    "Claim your free prize",        # Spam
    "Let's catch up tomorrow",      # Not Spam
    "Meeting schedule for today",   # Not Spam
    "Project update attached",      # Not Spam
    "You won a free vacation",      # Spam
    "Lunch at 1 PM?",               # Not Spam
]

labels = [1, 1, 1, 0, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Step 2: Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Step 4: Create and train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Test with a new email
new_email = ["Free tickets for you"]
new_email_transformed = vectorizer.transform(new_email)
prediction = model.predict(new_email_transformed)
print(f"\nPrediction for '{new_email[0]}':", "Spam" if prediction[0] == 1 else "Not Spam")
