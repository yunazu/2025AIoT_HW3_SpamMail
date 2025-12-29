
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

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
    print("Loading model and vectorizer...")
    try:
        model = joblib.load('spam_classifier_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        print("Model and vectorizer loaded successfully.")
    except FileNotFoundError:
        print("Error: Model or vectorizer file not found.")
        print("Please ensure 'spam_classifier_model.joblib' and 'tfidf_vectorizer.joblib' are in the current directory.")
        return

    print("\nEnter an SMS message to classify (type 'exit' to quit):")
    while True:
        sms_message = input("You: ")
        if sms_message.lower() == 'exit':
            break

        # Preprocess the input message
        processed_message = preprocess_text(sms_message)

        # Vectorize the preprocessed message
        message_tfidf = vectorizer.transform([processed_message])

        # Make prediction
        prediction = model.predict(message_tfidf)

        # Output result
        if prediction[0] == 1:
            print("Prediction: SPAM")
        else:
            print("Prediction: HAM")
        print("-" * 30)

if __name__ == '__main__':
    main()