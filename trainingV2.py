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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import SimpleRNN

lemmatizer = WordNetLemmatizer()

# Cargar intenciones desde un archivo JSON
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

def quitar_tildes(texto):
    texto_sin_tildes = unidecode.unidecode(texto)
    return texto_sin_tildes

def eliminar_palabras_de_parada(texto, idioma='spanish'):
    palabras = nltk.word_tokenize(texto)
    stop_words = set(stopwords.words(idioma))
    palabras_filtradas = [palabra for palabra in palabras if palabra.lower() not in stop_words]
    texto_procesado = ' '.join(palabras_filtradas)
    return texto_procesado

def convertir_a_singular(oracion):
    palabras = oracion.split()
    oracion_singular = [singularize(palabra) for palabra in palabras]
    return ' '.join(oracion_singular)

cont = 0

for intent in intents['intents']:
    for pattern in intent['patterns']:
        cont = cont + 1
        pattern = eliminar_palabras_de_parada(pattern, idioma='spanish')
        pattern = convertir_a_singular(pattern)
        texto_sin_tildes = quitar_tildes(pattern)
        pattern = texto_sin_tildes.lower()
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
            cont = cont + 1

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Definir una estructura para almacenar la conversación del usuario
conversation_history = {}

# Función para obtener o inicializar la historia de conversación de un usuario
def get_or_initialize_history(user_id):
    if user_id in conversation_history:
        return conversation_history[user_id]
    else:
        conversation_history[user_id] = []
        return conversation_history[user_id]

# Función para agregar un mensaje a la historia de conversación
def add_to_history(user_id, message):
    conversation_history[user_id].append(message)

# Función para preparar los datos de entrenamiento, incluyendo la historia de conversación
def prepare_training_data():
    training_data = []
    
    output_empty = [0]*len(classes)
    for document in documents:
        bag = [0] * len(words)
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

        # Incluir la historia de la conversación
        conversation_history_text = " ".join(get_or_initialize_history(document[1]))
        word_patterns = conversation_history_text + " " + " ".join(word_patterns)

        for word in word_patterns.split():
            if word in words:
                bag[words.index(word)] = 1
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training_data.append([bag, output_row])
    return training_data

training = prepare_training_data()
random.shuffle(training)

train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Obtén la longitud de los datos de entrada
input_length = len(train_x[0])

# Reformatea los datos de entrada a 3D
train_x = train_x.reshape(train_x.shape[0], 1, input_length)

# Crear el modelo con una capa RNN
model = Sequential()
model.add(SimpleRNN(128, input_shape=(train_x.shape[1], train_x.shape[2]), activation='relu'))  # train_x.shape[1] es la longitud de la secuencia
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
train_process = model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)

# Guardar el modelo
model.save("chatbot_model.h5", train_process)