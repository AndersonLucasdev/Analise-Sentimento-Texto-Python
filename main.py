## importações
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class SentimentAnalysis:
    ## inicializa as pastas
    def __init__(self, train_folder, test_folder):
        self.train_folder = train_folder
        self.test_folder = test_folder

    def load_data(self, folder_path):
        texts = []
        labels = []
        for label in ['pos', 'neg']:
            folder = os.path.join(folder_path, label)
            ## intera sobre os arquivos da pasta
            for filename in os.listdir(folder):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                    texts.append(file.read())
                labels.append(label)
        return texts, labels ## retorna as listas com os textos e os rótulos das críticas

    def train_model(self):
        ## Divide os dados (treinamento e teste)
        train_texts, train_labels = self.load_data(self.train_folder)
        test_texts, test_labels = self.load_data(self.test_folder)
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

        