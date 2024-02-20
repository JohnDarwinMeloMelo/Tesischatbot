import openai
openai.api_key = "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"
modelo = 'gpt-3.5-turbo'
pronpt = "como ayudar ah alguien que tiene vomito"
mensajes = [
    {"role":"system","content":"Dame una respuesta corta "},
    {"role":"system","content":"solo responde preguntas realciondas con el tema de violencia basada en genero si no es del tema de violencia basada en genero di que solo respondes de ese tema "},
    {"role":"user","content":pronpt}
]
respuesta = openai.ChatCompletion.create(
    model = modelo,
    messages = mensajes
)
print(respuesta['choices'][0]['message']['content'])
