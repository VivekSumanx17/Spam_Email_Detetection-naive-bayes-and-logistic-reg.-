import os
from collections import Counter
import pandas as pd
import numpy as np

# import modal from sklearn
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
import pickle

#Load dataset
file_path = "dataset/spam_ham_dataset.csv"
df = pd.read_csv(file_path)

emails = df['text'].tolist()

#build word dictionary
words = []

# convert every word as element in list
for email in emails:
    words += email.split(" ")

#  clean spaces from data
for i in range(len(words)):
    if not words[i].isalpha():
        words[i] = ""

# count the word frequency
words = [w for w in words if w != ""]
word_dict = Counter(words)

# remove uncommon words
word_dict = word_dict.most_common(3000)

# transform data into matrix form
features = []

for email in emails:
    temp = email.split()
    data = []
    for i in word_dict:
        data.append(temp.count(i[0]))
    features.append(data)

labels = df['label'].tolist()
labels = [1 if lbs == 'spam' else 0 for lbs in labels]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=9)

#Train classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

#Save classifier and word_dictionary
with open("Naive_bayes_model.pkl", "wb") as f:
    pickle.dump(classifier, f)

with open("word_dict.pkl", "wb") as f:
    pickle.dump(word_dict, f)

print("Model and word_dict saved successfully!")

# Accuracy check
from sklearn.metrics import accuracy_score

y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
