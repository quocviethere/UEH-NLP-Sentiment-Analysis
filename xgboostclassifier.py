from xgboost import XGBClassifier
import pickle
from sklearn.feature_extraction import DictVectorizer
import string
import pandas                   as pd
import matplotlib.pyplot        as plt
import re
import seaborn                  as sns
from collections                import Counter
import pickle
##########################################################################################
import nltk
from nltk.corpus                import stopwords, wordnet
from nltk                       import word_tokenize
from nltk                       import NaiveBayesClassifier, MaxentClassifier, classify
from nltk.metrics               import accuracy
##########################################################################################
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
##########################################################################################
import emot
import os
emot_obj = emot.emot()
##########################################################################################
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection    import train_test_split
##########################################################################################
from preprocess                 import preprocess_text
##########################################################################################


# read data
df = pd.read_csv('/Users/quocviet/Desktop/UEH_NLP/data/shopee_reviews.csv')

# preprocess data
df['preprocess_sentence'] = df['content'].apply(preprocess_text)

# word count
corpus = ' '.join(df['preprocess_sentence'])
tokens = word_tokenize(corpus)
word_count = Counter(tokens)

# tokenize
def tokenize(text):
    '''
    This function is used to tokenize text
    '''
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    return tokens

df['preprocess_sentence'] = df['preprocess_sentence'].apply(tokenize)

# create documents
documents = [(text, label) for text, label in zip(df['preprocess_sentence'], df['target'])]

def document_features(document):
    '''
    this function is used to create features
    '''
    features = {}
    for word in document:
        features[word] = True
    return features

# create featuresets
featuresets = [(document_features(d), c) for (d, c) in documents]

X, y = zip(*featuresets)

# Transform the list of dictionaries into a 2D array
vectorizer = DictVectorizer(sparse=False)
X_transformed = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_classifier.fit(X_train, y_train)

predicted_labels = xgb_classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels, average='weighted')
recall = recall_score(y_test, predicted_labels, average='weighted')
f1 = f1_score(y_test, predicted_labels, average='weighted')

print('XGBoost Sentiment Analysis Results')
print('----------------------------------')
print(f"Accuracy: \t {accuracy:.2f}")
print(f"Precision: \t {precision:.2f}")
print(f"Recall: \t {recall:.2f}")
print(f"F1-score: \t {f1:.2f}")

conf_matrix = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=xgb_classifier.classes_, 
            yticklabels=xgb_classifier.classes_,
            annot_kws={"size": 20})
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('True', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('XGBoost Confusion Matrix', fontsize=20)
plt.show()

filename = '/Users/quocviet/Documents/UEH-NLP-Sentiment-Analysis/xgb_sentiment_model.pkl'  
    
with open(filename, 'wb') as file:
    pickle.dump(xgb_classifier, file)

vectorizer_filename = '/Users/quocviet/Documents/UEH-NLP-Sentiment-Analysis/vectorizer.pkl'

os.makedirs(os.path.dirname(vectorizer_filename), exist_ok=True)

try:
    with open(vectorizer_filename, 'wb') as file:
        pickle.dump(vectorizer, file)
    print(f"Vectorizer successfully saved.")
except Exception as e:
    print(f"Error saving the vectorizer: {e}")