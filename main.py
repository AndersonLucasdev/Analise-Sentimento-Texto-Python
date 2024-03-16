## importações
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

class SentimentAnalysis:
    ## inicializa as pastas
    def __init__(self, train_folder, test_folder, classifier='nb'):
        """
        Inicializa a classe SentimentAnalysis.

        :param train_folder: Caminho para a pasta de treinamento.
        :param test_folder: Caminho para a pasta de teste.
        :param classifier: Algoritmo de classificação a ser utilizado (padrão: 'nb' para Naive Bayes).
        """
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.classifier = classifier
        self.texts_train = []
        self.labels_train = []
        self.texts_test = []
        self.labels_test = []
        self.model = None

    def load_data(self, folder_path):
        """
        Carrega os dados de texto e rótulos de sentimento a partir do caminho da pasta especificado.

        :param folder_path: Caminho para a pasta contendo os dados.
        :return: texts (list), labels (list)
        """
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
    
    def preprocess_data(self):
        """
        Carrega os dados de treinamento e teste.
        """
        self.texts_train, self.labels_train = self.load_data(self.train_folder)
        self.texts_test, self.labels_test = self.load_data(self.test_folder)

    def get_classifier(self):
        """
        Retorna o classificador selecionado com base no parâmetro 'classifier'.

        :return: Classificador sklearn
        """
        if self.classifier == 'nb':
            return MultinomialNB()
        elif self.classifier == 'svm':
            return SVC(kernel='linear')
        elif self.classifier == 'decision_tree':
            return DecisionTreeClassifier()
        else:
            raise ValueError("Algoritmo de classificação inválido!")

    def train_model(self):
        """
        Treina o modelo de análise de sentimento com base nos dados de treinamento.
        """
        train_texts, train_labels = self.load_data(self.train_folder)
        test_texts, test_labels = self.load_data(self.test_folder)
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels,
                                                                            test_size=0.2, random_state=42)

        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', self.get_classifier()), 
        ])
        text_clf.fit(train_texts, train_labels)

        val_predicted = text_clf.predict(val_texts)
        val_accuracy = accuracy_score(val_labels, val_predicted)
        print("Acurácia na validação:", val_accuracy)

        test_predicted = text_clf.predict(test_texts)
        test_accuracy = accuracy_score(test_labels, test_predicted)
        print("Acurácia nos dados de teste:", test_accuracy)

        self.model = text_clf

    def save_model(self, filename):
        """
        Salva o modelo treinado em um arquivo.

        :param filename: Nome do arquivo para salvar o modelo.
        """
        if self.model is not None:
            joblib.dump(self.model, filename)
            print("Modelo salvo com sucesso.")
        else:
            print("Nenhum modelo treinado para salvar.")

    def load_model(self, filename):
        """
        Carrega um modelo treinado de um arquivo.

        :param filename: Nome do arquivo contendo o modelo treinado.
        """
        try:
            self.model = joblib.load(filename)
            print("Modelo carregado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")

    def predict(self, texts):
        """
        Faz previsões de sentimento para os textos fornecidos.

        :param texts: Textos para os quais fazer previsões.
        :return: Previsões de sentimento.
        """
        if self.model is None:
            print("Nenhum modelo treinado disponível.")
            return None
        else:
            return self.model.predict(texts)
    
    def evaluate_model(self, texts, labels):
        """
        Avalia o modelo com base nos textos e rótulos fornecidos.

        :param texts: Textos para avaliação.
        :param labels: Rótulos verdadeiros dos textos.
        """
        if self.model is None:
            print("Nenhum modelo treinado disponível.")
            return
        else:
            predicted = self.model.predict(texts)
            print("Relatório de Classificação:")
            print(classification_report(labels, predicted))
            print("Matriz de Confusão:")
            print(confusion_matrix(labels, predicted))

if __name__ == "__main__":
    sentiment_analysis = SentimentAnalysis('aclImdb/train', 'aclImdb/test', classifier='svm')
    sentiment_analysis.train_model()
    sentiment_analysis.save_model('sentiment_model.pkl')