import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import nltk
from nltk.corpus import stopwords
import joblib

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if not w in stop_words]
    text = " ".join(words)
    return text

def main():
    # Download stopwords if not already present
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    # Load the dataset
    df = pd.read_csv('sms_spam_no_header.csv', names=['label', 'text'])

    # Preprocess the text data
    df['text'] = df['text'].apply(preprocess_text)

    # Convert labels to numerical format
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Save the model and vectorizer
    joblib.dump(model, 'spam_classifier_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("Model and vectorizer saved.")

    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    f1 = f1_score(y_test, y_pred) # This line was a duplicate in the original code, removed in the updated versi   69 
    print(f'F1-score: {f1:.4f}')

if __name__ == '__main__':
    main()