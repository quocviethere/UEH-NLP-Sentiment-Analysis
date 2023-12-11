import tkinter as tk
from tkinter import filedialog, messagebox
import pickle
import nltk
from preprocess import preprocess_text

root = tk.Tk()
root.title("Naive Bayes Classifier UI")

def predict_sentiment():
    user_text = entry.get()
    try:
        with open('maxent_classifier.pkl', 'rb') as model_file:
            maxent_classifier = pickle.load(model_file)

        tokenized_text = nltk.word_tokenize(preprocess_text(user_text))
        features = {word: True for word in tokenized_text}
        result = maxent_classifier.classify(features)

        if result == 1:
            sentiment_label.config(text="Positive Review", fg="green")
        else:
            sentiment_label.config(text="Negative Review", fg="red")
    except Exception as e:
        sentiment_label.config(text="Error: Model not found", fg="red")

label = tk.Label(root, text="Enter review:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

predict_button = tk.Button(root, text="Predict Sentiment", command=predict_sentiment)
predict_button.pack()

sentiment_label = tk.Label(root, text="")
sentiment_label.pack()

root.mainloop()