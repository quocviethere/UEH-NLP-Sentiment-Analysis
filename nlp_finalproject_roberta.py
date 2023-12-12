# -*- coding: utf-8 -*-
"""NLP-FinalProject-RoBERTa.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SYhO2FzJzzOChEz3_b_JHNJ2FDa6R4Wb
"""

!pip install -q transformers accelerate -U

from transformers import pipeline
import torch
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

sentiment_pipeline = pipeline("sentiment-analysis",model='siebert/sentiment-roberta-large-english')

data = ["I hate this product. It is very bad and doesn't worth it.",
        "I love the product. It's wonderful and bang for your buck."]
sentiment_pipeline(data)

# https://drive.google.com/file/d/1eqvo1mWn9SNio-7Crsx6s8TFXu1AO5JN/view?usp=sharing
!gdown 1eqvo1mWn9SNio-7Crsx6s8TFXu1AO5JN

class dataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

file_name = "/content/shopee_reviews.csv"
text_column = "content"

df = pd.read_csv(file_name)
df = df.sample(frac=1, random_state=42) #shuffle the data

# Split data into training and testing
split_index = int(0.8 * len(data))
train_data = df[:split_index]
test_data = df[split_index:]

pred_texts = train_data[text_column].dropna().astype('str').tolist()

model_name = "siebert/sentiment-roberta-large-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
trainer = Trainer(model=model)

tokenized_texts = tokenizer(pred_texts,truncation=True,padding=True)
pred_dataset = dataset(tokenized_texts)

predictions = trainer.predict(pred_dataset)

preds = predictions.predictions.argmax(-1)
labels = pd.Series(preds).map(model.config.id2label)
scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)

tokenized_texts = tokenizer(pred_texts,truncation=True,padding=True)
pred_dataset = dataset(tokenized_texts)

df = pd.DataFrame(list(zip(pred_texts,preds,labels,scores)), columns=['text','pred','label','score'])
df

# Load the test dataset and its corresponding labels
test_texts = test_data[text_column].dropna().astype('str').tolist()  # Replace 'text_column' with the appropriate column name from your dataset
test_labels = test_data['labels_column']  # Replace 'labels_column' with the column containing ground truth labels

# Tokenize the test texts
tokenized_test_texts = tokenizer(test_texts, truncation=True, padding=True)
test_dataset = dataset(tokenized_test_texts)

# Make predictions on the test dataset
test_predictions = trainer.predict(test_dataset)
test_preds = test_predictions.predictions.argmax(-1)

# Calculate evaluation metrics
test_accuracy = (test_preds == test_labels).mean()
test_f1 = f1_score(test_labels, test_preds, average='weighted')  # You need to import f1_score from sklearn.metrics
test_precision = precision_score(test_labels, test_preds, average='weighted')  # Import precision_score
test_recall = recall_score(test_labels, test_preds, average='weighted')  # Import recall_score

# Print the evaluation metrics
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-score: {test_f1:.4f}")