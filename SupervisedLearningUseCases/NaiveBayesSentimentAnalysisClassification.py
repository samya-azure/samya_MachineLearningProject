
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample movie reviews (positive = 1, negative = 0)
reviews = [
    "I love this movie",                      # Positive
    "This film was terrible",                # Negative
    "What a great performance",              # Positive
    "I hated the plot",                      # Negative
    "Amazing direction and acting",          # Positive
    "Not my type of movie",                  # Negative
    "The movie was fantastic",               # Positive
    "It was a waste of time",                # Negative
    "Brilliant and touching story",          # Positive
    "Disappointing and dull",                # Negative
    "This is an excellent film",             # Positive
    "What a terrible experience",            # Negative
    "Loved every bit of the story",          # Positive
    "The movie was dull and boring",         # Negative
    "Fantastic cinematography and direction",# Positive
    "I do not recommend this movie",         # Negative
]

# Corresponding sentiments
sentiments = [
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0
]

# Step 1: Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)
y = sentiments

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# Step 3: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 5: Try with new custom reviews
custom_reviews = [
    "This film was excellent",
    "What a boring movie",
    "I loved the acting",
    "Not my type of film"
]

custom_X = vectorizer.transform(custom_reviews)
predictions = model.predict(custom_X)

# Step 6: Display predictions
print("\nPredictions for new reviews:")
for review, pred in zip(custom_reviews, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Review: '{review}' → Sentiment: {sentiment}")
