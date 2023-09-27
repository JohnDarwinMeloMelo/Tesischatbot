import random
import json
import pickle
import numpy as np
import unidecode
import nltk
from nltk.stem import WordNetLemmatizer #Para pasar las palabras a su forma raíz
from nltk.corpus import stopwords
#Para crear la red neuronal
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

#from keras.optimizers import sgd_experimental
#from keras.optimizers import gradient_descent_v2 
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()


# Abre el archivo JSON con la codificación UTF-8
with open('intents.json', 'r', encoding='utf-8') as file:
    file_content = file.read()

# Carga el contenido del archivo JSON en un objeto Python
intents = json.loads(file_content)

# intents = json.loads(open('intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))  # Lista de stopwords en spanish

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

#Clasifica los patrones y las categorías
for intent in intents['intents']:
    
    for pattern in intent['patterns']:
        pattern = unidecode.unidecode(pattern.lower())
        word_list = nltk.word_tokenize(pattern)
         # Elimina las palabras de parada
        word_list = [word for word in word_list if word.lower() not in stop_words]
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Pasa la información a unos y ceros según las palabras presentes en cada categoría para hacer el entrenamiento
training = []
output_empty = [0]*len(classes)


for document in documents:
    bag = [0] * len(words)  # Inicializar bag con ceros
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in word_patterns:
        if word in words:
            bag[words.index(word)] = 1  # Cambiar el valor en la posición correspondiente
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

######################################################################## 
 
    
random.shuffle(training)
#training = np.array(training) 
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])
print(training) 

#Reparte los datos para pasarlos a la red
#train_x = list(training[:,0])
#train_y = list(training[:,1])

#Creamos la red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Creamos el optimizador y lo compilamos

#sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#Entrenamos el modelo y lo guardamos
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save("chatbot_model.h5", train_process)
