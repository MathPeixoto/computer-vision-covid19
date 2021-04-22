import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cnn import ConvNN
from metricas import Metricas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np
from glob import glob
import tensorflow as tf
import sys
import argparse
from tensorflow.keras.models import model_from_json

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

altura_imagem = 331
largura_imagem = 331
batch_size = 10
treino_path = 'dataset/training_set'
teste_path = 'dataset/test_set'
arquivo_modelo = 'pesos_covid.h5'
arquivo_modelo_json = 'covid.json'

quantidade_classes = len(glob(treino_path + '/*'))
quantidade_imagens_treino = len(glob(treino_path + '/*/*.png'))
quantidade_imagens_teste = len(glob(teste_path + '/*/*.png'))


def carregar_imagens():
    generator = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocess_input)

    treino_base = generator.flow_from_directory(treino_path,
                                                target_size=(altura_imagem, largura_imagem),
                                                batch_size=batch_size)
    teste_base = generator.flow_from_directory(teste_path,
                                               target_size=(altura_imagem, largura_imagem),
                                               batch_size=batch_size)
    return treino_base, teste_base


def callbacks():
    lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=3, verbose=1)
    early_stopper = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto")
    checkpointer = ModelCheckpoint(arquivo_modelo, monitor="val_loss", verbose=1, save_best_only=True)

    return lr_reducer, early_stopper, checkpointer


def modelo_salvo():
    arquivo = open('covid.json', 'r')
    estrutura_rede = arquivo.read()
    arquivo.close()

    modelo = model_from_json(estrutura_rede)
    modelo.load_weights('pesos_covid.h5')

    return modelo


def novo_modelo():
    treino_base, teste_base = carregar_imagens()
    lr_reducer, early_stopper, checkpointer = callbacks()
    cnn = ConvNN(altura_imagem, largura_imagem)
    metricas = Metricas()

    modelo = cnn.cria_cnn(quantidade_classes)
    historico = modelo.fit(treino_base,
                           steps_per_epoch=quantidade_imagens_treino // batch_size,
                           epochs=100,
                           batch_size=batch_size,
                           validation_data=teste_base,
                           validation_steps=quantidade_imagens_teste // batch_size,
                           callbacks=[lr_reducer, early_stopper, checkpointer])

    cnn.salvar_modelo(modelo, arquivo_modelo_json)
    metricas.plota_historico_modelo(historico)

    return modelo, historico


def get_diagnostico(modelo, image_test):
    image_test = image.img_to_array(image_test)
    image_test = np.expand_dims(image_test, axis=0)

    previsao = modelo.predict(image_test)
    arg = np.argmax(previsao)

    diagnostico = ['covid', 'normal', 'viral pneumonia']
    print(diagnostico[arg])


def main():
    parser = argparse.ArgumentParser(description='tcc')
    parser.add_argument('--default')
    parser.add_argument('--path')
    args = parser.parse_args()

    image_test = image.load_img(args.path,
                                target_size=(altura_imagem, largura_imagem))

    if args.default == 'y':
        modelo = modelo_salvo()

    else:
        modelo, historico = novo_modelo()

    get_diagnostico(modelo, image_test)

    return 0


if __name__ == '__main__':
    sys.exit(main())
