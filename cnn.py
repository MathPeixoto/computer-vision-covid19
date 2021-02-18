from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

class ConvNN:
    
    def __init__(self, altura_imagem, largura_imagem):
        self.altura_imagem = altura_imagem
        self.largura_imagem = largura_imagem
        self.IMAGE_SIZE = [altura_imagem, largura_imagem, 3]

    def cria_cnn(self, quantidade_classes) :
        #   Adiciona uma camada de pré-processamento à frente da ResNet50
        resnet = ResNet50(input_shape=self.IMAGE_SIZE, weights='imagenet', include_top=False)
        
        #   Loop necessário para não treinar as camadas já treinadas anteriormente.
        #   Essa técnica é chamada de Transfer Learning 
        for layer in resnet.layers:
            layer.trainable = False
            
        x = Flatten()(resnet.output)
        output = Dense(units=quantidade_classes, activation='softmax')(x)
        #create a custom model from ResNet50
        modelo = Model(inputs=resnet.input, outputs=output)
        modelo.summary()
        modelo.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        
        return modelo
    
    def salvar_modelo(self, modelo, arquivo_modelo_json):
        # Salvando a arquitetura do modelo em um arquivo JSON
        model_json = modelo.to_json()
        with open(arquivo_modelo_json, "w") as json_file:
          json_file.write(model_json)
