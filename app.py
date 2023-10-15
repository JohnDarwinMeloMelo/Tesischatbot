# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request

from heyoo import WhatsApp
import json
import random
import json
import pickle
import numpy as np
import torch
import nltk
import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.models import load_model
import os
from langdetect import detect 
nltk.download('stopwords')
app = Flask(__name__)

# Ruta para recibir las solicitudes de WhatsApp
@app.route("/webhook/", methods=["POST", "GET"])
def webhook_whatsapp():
    if request.method == "GET":
        if request.args.get('hub.verify_token') == "chatbotvbg":
            return request.args.get('hub.challenge')
        else:
            return "Error de autenticacion."

    data = request.get_json()

    telefonoCliente = None
    mensaje = None
    idWA = None
    timestamp = None

    if 'entry' in data and data['entry']:
        entry = data['entry'][0]
        if 'changes' in entry and entry['changes']:
            change = entry['changes'][0]
            if 'value' in change and 'messages' in change['value'] and change['value']['messages']:
                telefonoCliente = change['value']['messages'][0]['from']
                mensaje = change['value']['messages'][0]['text']['body']
                idWA = change['value']['messages'][0]['id']
                timestamp = change['value']['messages'][0]['timestamp']

    if mensaje is not None:
        
        
        lemmatizer = WordNetLemmatizer()

        
        # Abre el archivo JSON con la codificación UTF-8
        with open('intents.json', 'r', encoding='utf-8') as file:
            file_content = file.read()

        # Carga el contenido del archivo JSON en un objeto Python
        intents = json.loads(file_content)
        
        
        #intents = json.loads(open('intents.json').read())
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
        model = load_model('chatbot_model.h5')

        #Pasamos las palabras de oración a su forma raíz
        def clean_up_sentence(sentence):
            sentence_words = nltk.word_tokenize(sentence)
            sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
            
            mensaje2= " ".join(sentence_words)
    
            print("mensaje 2: "+ mensaje2)
            return sentence_words

        #Convertimos la información a unos y ceros según si están presentes en los patrones
        def bag_of_words(sentence):
            
            sentence_words = clean_up_sentence(sentence)
            
            bag = [0]*len(words)
            for w in sentence_words:
                for i, word in enumerate(words):
                    if word == w:
                        bag[i]=1
            print(bag)
            return np.array(bag)

        
        #Predecimos la categoría a la que pertenece la oración
        def predict_class(sentence):
            bow = bag_of_words(sentence)
            
            
            ####################
            has_ones = np.any(bow)

            if has_ones:
                res = model.predict(np.array([bow]))[0]
                max_index = np.where(res ==np.max(res))[0][0]
                category = classes[max_index]
            else:
                category = "desconocido"
            print(bow)        
            print(category)
            return category
        

        
        
        #Obtenemos una respuesta aleatoria
        def get_response(tag, intents_json):
            list_of_intents = intents_json['intents']
            result = ""
            for i in list_of_intents:
                if i["tag"]==tag:
                    result = random.choice(i['responses'])
                    break
            return result

        #-------------------------------------------
        #Dectectar palabra mal escrita ----------------------------------------------------------------

        
        
        #----------------------------------------INICIO DE MENSAJE 
        
        
        
        mensaje = unidecode.unidecode(mensaje.lower())
        print("mensaje 1: "+mensaje)
        
        ints = predict_class(mensaje)
   

        
        
        
       
        
        respuesta = get_response(ints, intents)
        
        
    

        with open("texto.txt", "w", encoding="utf-8") as f:
            f.write(respuesta)



        enviar(telefonoCliente, respuesta)

        return jsonify({"status": "success"}, 200)
    
    else:
        # En caso de que no haya mensaje, puedes retornar una respuesta de error
        return jsonify({"status": "error", "message": "No se recibió un mensaje válido"}, 400)

def enviar(telefonoRecibe, respuesta):
    token = 'EAALZBZB8exxfkBO1dKOrQIR2e3nZCNK9ZANZCo7VSkInr1aepGYWrvM3jSgjGnnwejZAkXaFrwNwRn4zzZCSP8hLhzr9QcSh3bAEKMbxxkVX8msLPGqDZAAtpeGhZBvQZCv5JVLA0MfJQQyRPu8Rt2NEkMoZCXIMs4qqxAHmGMvEc8k7PnaLO0EiVF7PayuTKRZC'  # Reemplaza con tu token de WhatsApp
    idNumeroTelefono = '117836178073425'  # Reemplaza con tu ID de número de teléfono

    mensajeWa = WhatsApp(token, idNumeroTelefono)
    telefonoRecibe = telefonoRecibe.replace("521", "52")
    
    mensajeWa.send_message(respuesta, telefonoRecibe)

if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False
    
    app.run(host='0.0.0.0', port=5000)
