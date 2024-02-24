import numpy as np
import os
import sklearn as sk

def carregar_dados(folder_path):
    texts = []
    labels = []
    for label in ['pos', 'neg']:
        folder = os.path.join(folder_path, label)
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
            labels.append(label)
    return texts, labels