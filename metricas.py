import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

class Metricas:
    
    def __init__(self):
        pass
    
    # Gerando gráfico da melhora em cada etapa do treinamento
    def plota_historico_modelo(self, historico_modelo):
        fig, axs = plt.subplots(1, 2, figsize=(15,5))
        
        axs[0].plot(range(1, len(historico_modelo.history["accuracy"]) + 1), 
                    historico_modelo.history["accuracy"], "r")
        axs[0].plot(range(1, len(historico_modelo.history["val_accuracy"]) + 1),
                    historico_modelo.history["val_accuracy"], "b")
        axs[0].set_title("Acurácia do modelo")
        axs[0].set_ylabel("Acurácia")
        axs[0].set_xlabel("Épocas")
        axs[0].set_xticks(np.arange(1, len(historico_modelo.history["accuracy"]) + 1),
                          len(historico_modelo.history["accuracy"]) / 10)
        axs[0].legend(["Acurácia no treinamento", "Acurácia na validação"], loc = "best")
        
        
        axs[1].plot(range(1, len(historico_modelo.history["loss"]) + 1),
                    historico_modelo.history["loss"], "r")
        axs[1].plot(range(1, len(historico_modelo.history["val_loss"]) + 1),
                    historico_modelo.history["val_loss"], "b")
        axs[1].set_title("Loss do modelo")
        axs[1].set_ylabel("Loss")
        axs[1].set_xlabel("Épocas")
        axs[1].set_xticks(np.arange(1, len(historico_modelo.history["loss"]) + 1),
                          len(historico_modelo.history["loss"]) / 10)
        axs[1].legend(["Loss de treinamento", "Loss de validação"], loc = "best")
        fig.savefig("historico_modelo.png")

    # Gerando a Matriz de Confusão
    def plotar_matriz_confusao(self, classe_verdadeira, classe_prevista):
        cm = confusion_matrix(classe_verdadeira, classe_prevista)
        diagnostico = ['covid', 'normal', 'viral pneumonia']
        titulo = "Matriz de confusao"
        print(cm)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(titulo)
        plt.colorbar()
        tick_marcks = np.arange(len(diagnostico))
        plt.xticks(tick_marcks, diagnostico, rotation = 45)
        plt.yticks(tick_marcks, diagnostico, rotation = 45);
        
        fmt = "d"
        thresh = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                   color="white" if cm[i,j] > thresh else "black")
        
        plt.ylabel("Classificaçãão Correta")
        plt.xlabel("Prediçãão")
        plt.savefig("matriz_confusao.png")
