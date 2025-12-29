import streamlit as st
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

# Load the saved model and vectorizer
try:
    model = joblib.load('spam_classifier_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'spam_classifier_model.joblib' and 'tfidf_vectorizer.joblib' are in the current directory.")
    st.stop()

# Streamlit App
st.title("SMS Spam Classifier")
st.write("Enter an SMS message below to classify it as Spam or Ham.")

# Text input
user_input = st.text_area("Enter your message here:")

# Classify button
if st.button("Classify"):
    if user_input:
        # Preprocess the input
        processed_input = preprocess_text(user_input)

        # Vectorize the input
        vectorized_input = vectorizer.transform([processed_input])

        # Make prediction
        prediction = model.predict(vectorized_input)

        # Display result
        if prediction[0] == 1:
            st.error("This message is classified as: **SPAM**")
        else:
            st.success("This message is classified as: **HAM**")
    else:
        st.warning("Please enter a message to classify.")