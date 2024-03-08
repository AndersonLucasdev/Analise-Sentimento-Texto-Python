## importações
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

class SentimentAnalysis:
    ## inicializa as pastas
    def __init__(self, train_folder, test_folder, classifier='nb'):
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.classifier = classifier

    

    def load_data(self, folder_path):
        try:
            texts = []
            labels = []
            for label in ['pos', 'neg']:
                folder = os.path.join(folder_path, label)
                for filename in os.listdir(folder):
                    with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                        texts.append(file.read())
                    labels.append(label)
            return texts, labels ## retorna as listas com os textos e os rótulos das críticas
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")
            return None, None 

    def train_model(self):
        ## Divide os dados (treinamento e teste)
        try:
            train_texts, train_labels = self.load_data(self.train_folder)
            test_texts, test_labels = self.load_data(self.test_folder)
            if train_texts is None or test_texts is None:
                return
            
            train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

            text_clf = Pipeline([
                ('vect', CountVectorizer()),
                ('clf', MultinomialNB()),
            ])
            text_clf.fit(train_texts, train_labels)

            val_predicted = text_clf.predict(val_texts)
            val_accuracy = accuracy_score(val_labels, val_predicted)
            print("Acurácia na validação:", val_accuracy)

            test_predicted = text_clf.predict(test_texts)
            test_accuracy = accuracy_score(test_labels, test_predicted)
            print("Acurácia nos dados de teste:", test_accuracy)
        except Exception as e:
            print(f"Erro durante o treinamento do modelo: {e}")


if __name__ == "__main__":
    sentiment_analysis = SentimentAnalysis('aclImdb/train', 'aclImdb/test')
    sentiment_analysis.train_model()