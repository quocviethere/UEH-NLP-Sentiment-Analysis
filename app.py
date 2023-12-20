import tkinter as tk
from tkinter import ttk  # ttk is needed for the Combobox widget
import pickle
import nltk
from preprocess import preprocess_text

root = tk.Tk()
root.title("Sentiment Analysis UI")

def predict_sentiment():
    user_text = entry.get()
    selected_model = model_choice.get()

    try:
        if selected_model == "Maxent" or selected_model == "Naive Bayes":
            model_filename = f'models/{selected_model.lower().replace(" ", "_")}_classifier.pkl'
            with open(model_filename, 'rb') as model_file:
                model = pickle.load(model_file)

            tokenized_text = nltk.word_tokenize(preprocess_text(user_text))
            features = {word: True for word in tokenized_text}

            if selected_model in ["Maxent", "Naive Bayes"]:
                result = model.classify(features)

        elif selected_model == "XGBoost":
            model_filename = 'models/xgb_sentiment_model.pkl'
            vectorizer_filename = 'models/vectorizer.pkl'
            
            with open(model_filename, 'rb') as model_file, open(vectorizer_filename, 'rb') as vectorizer_file:
                model = pickle.load(model_file)
                vectorizer = pickle.load(vectorizer_file)

            tokenized_text = nltk.word_tokenize(preprocess_text(user_text))
            features = {word: True for word in tokenized_text}
            transformed_features = vectorizer.transform([features])

            result = model.predict(transformed_features)[0]

        else:
            sentiment_label.config(text="Select a valid model", fg="red")
            return

        if result == 1:
            sentiment_label.config(text="Positive Review", fg="green")
        else:
            sentiment_label.config(text="Negative Review", fg="red")

    except Exception as e:
        sentiment_label.config(text=f"Error: {e}", fg="red")

# UI Elements
label = tk.Label(root, text="Enter review:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

# Model selection combobox
model_label = tk.Label(root, text="Choose a model:")
model_label.pack()

model_choice = ttk.Combobox(root, values=["Maxent", "Naive Bayes", "XGBoost"])
model_choice.pack()

predict_button = tk.Button(root, text="Predict Sentiment", command=predict_sentiment)
predict_button.pack()

sentiment_label = tk.Label(root, text="")
sentiment_label.pack()

root.mainloop()
