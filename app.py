import spacy
import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, send
from http.client import responses

import bs4 as bs
import urllib.request
import re
import nltk
import numpy as np
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

# python3 -m spacy download pt


# Etapa 2: Carregamento e pré-processamento da base de dados
dados = urllib.request.urlopen(
    'https://en.wikipedia.org/wiki/Artificial_intelligence')
dados = dados.read()
dados_html = bs.BeautifulSoup(dados, 'lxml')
paragrafos = dados_html.find_all('p')

conteudo = ''
for p in paragrafos:
    conteudo += p.text

conteudo = conteudo.lower()
lista_sentencas = nltk.sent_tokenize(conteudo)

pln = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS


def preprocessamento(texto):
    texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)
    texto = re.sub(r" +", ' ', texto)
    documento = pln(texto)
    lista = []
    for token in documento:
        lista.append(token.lemma_)
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
    lista = ' '.join([str(elemento)
                     for elemento in lista if not elemento.isdigit()])

    return lista


lista_sentencas_preprocessada = []
for i in range(len(lista_sentencas)):
    lista_sentencas_preprocessada.append(preprocessamento(lista_sentencas[i]))

# Etapa 3: Frases de boas-vindas
textos_boasvindas_entrada = ("hey", "hello", "hi", "whats up", "how is going",)
textos_boasvindas_respostas = [
    "hey", "hello :)", "welcome", "it's a pleasure have you here", "hi :)", 
    "I'am Artificial Inteligence thanks!"]


def responder_saudacao(texto):
    for palavra in texto.split():
        if palavra.lower() in textos_boasvindas_entrada:
            return random.choice(textos_boasvindas_respostas)


# Etapa 4: Função para o chatbot responder o usuário
def responder(texto_usuario):

    resposta_chatbot = ''
    lista_sentencas_preprocessada.append(texto_usuario)

    tfidf = TfidfVectorizer()
    palavras_vetorizadas = tfidf.fit_transform(lista_sentencas_preprocessada)

    similaridade = cosine_similarity(
        palavras_vetorizadas[-1], palavras_vetorizadas)

    indice_sentenca = similaridade.argsort()[0][-2]
    vetor_similar = similaridade.flatten()
    vetor_similar.sort()
    vetor_encontrado = vetor_similar[-2]

    if(vetor_encontrado == 0):
        resposta_chatbot = resposta_chatbot + "Sorry repeat please!!"
        return resposta_chatbot
    else:

        print(texto_usuario)
        if(texto_usuario == "chat ai"):
            print("yes")
            resposta_chatbot = ''
            return resposta_chatbot
        else:
            resposta_chatbot = resposta_chatbot + \
                lista_sentencas[indice_sentenca]
            return resposta_chatbot


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
socketio = SocketIO(app, cors_allowed_origins="*", port=5000)


# Etapa 6: Função para retornar as respostas
@socketio.on('message')
def menssagens(msg):

    print('Message: ' + msg)
    send(msg, broadcast=True)

    resposta = ''
    texto_usuario = msg
    texto_usuario = texto_usuario.lower()
    if(responder_saudacao(texto_usuario) != None):
        resposta = responder_saudacao(texto_usuario)
        return send(resposta, broadcast=True)

    else:
        resposta = responder(preprocessamento(texto_usuario))
        lista_sentencas_preprocessada.remove(preprocessamento(texto_usuario))
    return send(resposta, broadcast=True)


if __name__ == '__main__':
    socketio.run(app)
