# -*- coding: utf-8 -*-
import random
import json
import pickle
import numpy as np
import unidecode
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pattern.es import singularize
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

lemmatizer = WordNetLemmatizer()

with open('intents.json', 'r', encoding='utf-8') as file:
    file_content = file.read()

intents = json.loads(file_content)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

def remove_accents(text):
    text_no_accents = unidecode.unidecode(text)
    return text_no_accents

def remove_stop_words(text, language='spanish'):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words(language))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text_processed = ' '.join(filtered_words)
    return text_processed

def to_singular(sentence):
    words = sentence.split()
    singular_sentence = [singularize(word) for word in words]
    return ' '.join(singular_sentence)

# Classify patterns and categories
for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = remove_stop_words(pattern, language='spanish')
        pattern = to_singular(pattern)
        text_no_accents = remove_accents(pattern)
        pattern = text_no_accents.lower()
        print("pattern: " + pattern)
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
            print("tag: " + intent["tag"] + "\n\n")




words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Convert the information into 1s and 0s based on the words present in each category for training
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [0] * len(words)
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in word_patterns:
        if word in words:
            bag[words.index(word)] = 1
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])
print(training)

# Create the neural network model
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Train the model
print("Training...")
train_process = model.fit(X_train, y_train, epochs=500, batch_size=3, verbose=1, validation_data=(X_test, y_test))

# Save the model
model.save("chatbot_model.h5")
