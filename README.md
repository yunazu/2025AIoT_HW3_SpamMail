# SMS Spam Classification

This project implements a machine learning model to classify SMS messages as either "spam" or "ham" (not spam). The classification is performed using a Logistic Regression model trained on a publicly available dataset.      

## Project Structure

*   `sms_spam_no_header.csv`: The dataset used for training and evaluation.
*   `spam_classifier.py`: Python script to preprocess the data, train the Logistic Regression model, evaluate it      performance, and save the trained model and TF-IDF vectorizer.
*   `spam_classifier_model.joblib`: The saved trained Logistic Regression model.
*   `tfidf_vectorizer.joblib`: The saved trained TF-IDF vectorizer.
*   `predict_spam.py`: Python script to load the saved model and vectorizer, and predict the class of new SMS   messages provided by the user.

## Setup Instructions

1.  **Clone the repository or download the project files.**
    (Assuming you have the project files in your local directory)

2.  **Install Python dependencies:**
    Open your terminal or command prompt in the project directory and run:pip install pandas scikit-learn 

3.  **Download NLTK stopwords (if not already downloaded):**
    The `spam_classifier.py` script will attempt to download NLTK stopwords if they are not present. Ensure you have an internet connection when you first run it.

4.  **Dataset:**
    Ensure the `sms_spam_no_header.csv` file is present in the root of your project directory. If you need to   download it, you can find it at:
    `https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads      aster/Chapter03/datasets/sms_spam_no_header.csv`

## Usage

### 1. Train and Evaluate the Model

Run the `spam_classifier.py` script to train the model, evaluate its performance, and save the trained model and      vectorizer.
on spam_classifier.

This script will output the model's performance scores and save `spam_classifier_model.joblib` and  `tfidf_vectorizer.joblib` in your project directory.

### Model Performance:

After training, the model achieved the following performance metrics:
*   **Accuracy:** 0.9614
*   **Precision:** 0.9538
*   **Recall:** 0.7702
*   **F1-score:** 0.8522

### 2. Predict New SMS Messages

Once the model and vectorizer are saved, you can use `predict_spam.py` to classify new SMS messages.
on predict_spam.

The script will prompt you to enter an SMS message. Type your message and press Enter. Type 'exit' to quit the  prediction tool.

## How it Works

The `spam_classifier.py` script performs the following steps:
1.  **Loads Data:** Reads the `sms_spam_no_header.csv` dataset.
2.  **Text Preprocessing:** Converts text to lowercase, removes punctuation, and removes common English stopword    9 3.  **Feature Extraction:** Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numeric      features.
4.  **Model Training:** Trains a Logistic Regression model on the processed data.
5.  **Evaluation:** Assesses the model's performance using accuracy, precision, recall, and F1-score.
6.  **Saves Model:** Saves the trained `LogisticRegression` model and `TfidfVectorizer` to `.joblib` files for  later use.

The `predict_spam.py` script:
1.  Loads the saved model and vectorizer.
2.  Takes user input for an SMS message.
3.  Applies the same preprocessing and vectorization steps to the input message.
4.  Uses the loaded model to predict if the message is "spam" or "ham" and prints the result.