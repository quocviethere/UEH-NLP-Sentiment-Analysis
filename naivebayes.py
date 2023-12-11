import string
import pandas                   as pd
import matplotlib.pyplot        as plt
import re
import seaborn                  as sns
from collections                import Counter
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
emot_obj = emot.emot()
##########################################################################################
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection    import train_test_split
##########################################################################################
from preprocess                 import preprocess_text
##########################################################################################

df = pd.read_csv('/Users/quocviet/Desktop/UEH_NLP/data/shopee_reviews.csv')

df['preprocess_sentence'] = df['content'].apply(preprocess_text)

corpus = ' '.join(df['preprocess_sentence'])
tokens = word_tokenize(corpus)
word_count = Counter(tokens)

def tokenize(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    return tokens

df['preprocess_sentence'] = df['preprocess_sentence'].apply(tokenize)

documents = [(text, label) for text, label in zip(df['preprocess_sentence'], df['target'])]

def document_features(document):
    features = {}
    for word in document:
        features[word] = True
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]


"""## Naive Bayes Classifier"""

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

# fit naive bayes classifier to the train set
nb_classifier = NaiveBayesClassifier.train(train_set)

# predict
true_labels = [label for (_, label) in test_set]
predicted_labels = [nb_classifier.classify(features) for (features, _) in test_set]

# evaluation
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"Accuracy: \t {accuracy:.2f}")
print(f"Precision: \t {precision:.2f}")
print(f"Recall: \t {recall:.2f}")
print(f"F1-score: \t {f1:.2f}")

# confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=nb_classifier.labels(),
            yticklabels=nb_classifier.labels(),
            annot_kws={"size": 20})
plt.xlabel('Predicted',fontsize=20)
plt.ylabel('True',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Naive Bayes Confusion Matrix',fontsize=20)
plt.show()