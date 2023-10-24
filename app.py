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
import mysql.connector
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.models import load_model
import os
from langdetect import detect 

from pattern.es import singularize
nltk.download('stopwords')
app = Flask(__name__)
mensaje_global=""
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
            return ' '.join(sentence_words)

        #Convertimos la información a unos y ceros según si están presentes en los patrones
        
        def bag_of_words(sentence):
            global general_mensaje
            sentence_words = []
            sentence_word = clean_up_sentence(sentence)
            sentence_word = eliminar_palabras_de_parada(sentence_word, idioma='spanish')
            
            sentence_word = convertir_a_singular(sentence_word)
            sentence_word = unidecode.unidecode(sentence_word.lower())
            
            general_mensaje=sentence_word
            #-------------------------------------------------VALIDACION _-__________
            #get_last_record_by_telefono()
            print("mensaje 1: "+general_mensaje)
            word_list = nltk.word_tokenize(sentence_word)
            sentence_words.extend(word_list)
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
        #base de datos ----------------------------------------------------------------
        def insert_data_to_database(idWA, mensaje, respuesta, timestamp, telefonoCliente):
            
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database='chatvbg'
            )
            mycursor = mydb.cursor()
            query = "SELECT count(id) AS cantidad FROM registro WHERE id_wa='" + idWA + "';"
            mycursor.execute(query)
            cantidad, = mycursor.fetchone()
            cantidad = str(cantidad)
            cantidad = int(cantidad)
            if cantidad == 0:
                sql = ("INSERT INTO registro" +
                    "(mensaje_recibido, mensaje_enviado, id_wa, timestamp_wa, telefono_wa) VALUES " +
                    "('" + mensaje + "','" + respuesta + "','" + idWA + "','" + timestamp + "','" + telefonoCliente + "');")
                mycursor.execute(sql)
                mydb.commit()
        
        def get_last_record_by_telefono():
            try:
                mydb = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",
                    database='chatvbg'
                )
                mycursor = mydb.cursor()

                # Consulta SQL para obtener el último registro por número de teléfono
                query = "SELECT * FROM registro WHERE telefono_wa = %s ORDER BY timestamp_wa DESC LIMIT 1;"
                
                # Reemplaza 'telefono_de_cliente' con el número de teléfono del cliente
                telefono_de_cliente = telefonoCliente  # Reemplaza con el número de teléfono real

                mycursor.execute(query, (telefono_de_cliente,))
                result = mycursor.fetchone()
                
                if result:
                    # El resultado contiene los campos del registro
                    id, mensaje_recibido, mensaje_enviado, id_wa, timestamp_wa, telefono_wa = result

                    print(f'ID: {id}')
                    print(f'Mensaje Recibido: {mensaje_recibido}')
                    print(f'Mensaje Enviado: {mensaje_enviado}')
                    print(f'ID de WhatsApp: {id_wa}')
                    print(f'Timestamp de WhatsApp: {timestamp_wa}')
                    print(f'Teléfono del Cliente: {telefono_wa}')
                else:
                    print("No se encontraron registros para el número de teléfono especificado.")

            except Exception as e:
                print(f"Error al consultar la base de datos: {str(e)}")
            finally:
                mydb.close()
        
        
        
        
        #----------------------------------------INICIO DE MENSAJE 
        def eliminar_palabras_de_parada(texto, idioma='spanish'):
            palabras = nltk.word_tokenize(texto)
            stop_words = set(stopwords.words(idioma))
            palabras_filtradas = [palabra for palabra in palabras if palabra.lower() not in stop_words]
            #palabras_filtradas = [plural_a_singular(palabra) if palabra.lower() not in stop_words else palabra for palabra in palabras]
            texto_procesado = ' '.join(palabras_filtradas)
            return texto_procesado
        
        def plural_a_singular(palabra):
            if palabra.endswith("es"):
                palabra_singular = palabra[:-2]  # Elimina la "s" final
                return palabra_singular
            elif palabra.endswith("s"):
                palabra_singular = palabra[:-1]  # Elimina "es" final
                return palabra_singular
            else:
                return palabra
        

        def convertir_a_singular(oracion):
            palabras = oracion.split()
            oracion_singular = [singularize(palabra) for palabra in palabras]
            return ' '.join(oracion_singular)

        # Ejemplo de uso:
        
        
        
        ints = predict_class(mensaje)
        respuesta = get_response(ints, intents)
        
        
        
        global general_mensaje
        insert_data_to_database(idWA, general_mensaje, respuesta, timestamp, telefonoCliente)
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
