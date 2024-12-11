import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

# Classes do dataset CIFAR-10
CLASSES = ['avião', 'automóvel', 'pássaro', 'gato', 'veado', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

# Carregar o modelo salvo
model = tf.keras.models.load_model('ia/best_model.keras')

# Carregar o dataset CIFAR-10
(_, _), (x_test, y_test) = datasets.cifar10.load_data()

# Normalizar os dados
x_test = x_test.astype('float32') / 255.0

def predict_image(index):
    """
    Exibe uma imagem do conjunto de teste, faz a previsão e verifica se está correta.
    """
    # Fazer previsão
    prediction = model.predict(x_test[index:index+1])
    predicted_label = np.argmax(prediction)

    # Verifica se está correto
    is_correct = "CORRETO" if predicted_label == y_test[index][0] else "INCORRETO"

    # Exibe a imagem com os rótulos
    plt.imshow(x_test[index])
    plt.title(f"Verdadeiro: {CLASSES[y_test[index][0]]} | Previsto: {CLASSES[predicted_label]} ({is_correct})")
    plt.axis('off')
    plt.show()

print("# Iniciando testador")
while True:
    print("## Para sair aperte CTRL+C após fechar a imagem (as vezes pode ser necessario digitar algo antes de fechar)")
    # Solicitar índice ao usuário
    try:
        index = int(input(f"# Escolha um índice entre 0 e {len(x_test) - 1}: "))
        if 0 <= index < len(x_test):
            predict_image(index)
        else:
            print("## Índice fora do intervalo!")
    except ValueError:
        print("## Por favor, insira um número válido.")
    except KeyboardInterrupt:
        print("# Fim do programa")
        break
