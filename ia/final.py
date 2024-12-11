# Importação das bibliotecas necessárias
from dependecies import install_dependecies
install_dependecies()


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from ia.utils import Dataset  # Classe Dataset deve estar definida no arquivo utils
from sklearn.metrics import classification_report, confusion_matrix

# Classes do dataset CIFAR-10 (em português)
CLASSES = ['avião', 'automóvel', 'pássaro', 'gato', 'veado', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

# Carregamento e preprocessamento dos dados do CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
train = Dataset(x_train, y_train)
test = Dataset(x_test, y_test)

# Exibe os formatos dos dados para validação
print(f"Formato do conjunto de treinamento: {train.x.shape}, {train.y.shape}")
print(f"Formato do conjunto de teste: {test.x.shape}, {test.y.shape}")

# Normalização dos dados para o intervalo [0, 1]
train.x = train.x.astype('float32') / 255.0
test.x = test.x.astype('float32') / 255.0

# Definição do modelo de rede neural convolucional (CNN)
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compila o modelo definindo o otimizador, a função de perda e a métrica de avaliação
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Configura os callbacks para monitorar o treinamento
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Treina o modelo utilizando os callbacks
history = cnn.fit(
    train.x, train.y,
    validation_data=(test.x, test.y),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, model_checkpoint]
)

# Avaliação do modelo no conjunto de teste
test_loss, test_acc = cnn.evaluate(test.x, test.y)
print(f"Acurácia no conjunto de teste: {test_acc}, Perda no conjunto de teste: {test_loss}")

# Visualiza os resultados do treinamento
# Gráfico da perda ao longo das epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Perda no Treinamento')
plt.plot(history.history['val_loss'], label='Perda na Validação')
plt.title('Perda ao longo das Épocas')
plt.xlabel('Epochs')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Gráfico da acurácia ao longo das epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Acurácia no Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia na Validação')
plt.title('Acurácia ao longo das Épocas')
plt.xlabel('Epochs')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Previsões no conjunto de teste
predictions = cnn.predict(test.x)
predicted_classes = np.argmax(predictions, axis=1)

# Visualiza algumas previsões com comparação entre os rótulos reais e previstos
num_images = 5
random_indices = np.random.choice(len(test.x), num_images, replace=False)
plt.figure(figsize=(5, 15))

for i, idx in enumerate(random_indices):
    plt.subplot(num_images, 1, i + 1)
    plt.imshow(test.x[idx])
    plt.axis('off')
    true_label = CLASSES[test.y[idx][0]]
    predicted_label = CLASSES[predicted_classes[idx]]
    plt.title(f"Verdadeiro: {true_label} | Previsto: {predicted_label}", fontsize=12)

plt.tight_layout()
plt.show()

# Gera matriz de confusão e exibe com Matplotlib
conf_matrix = confusion_matrix(test.y, predicted_classes)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()
tick_marks = np.arange(len(CLASSES))
plt.xticks(tick_marks, CLASSES, rotation=45)
plt.yticks(tick_marks, CLASSES)
plt.xlabel('Classes Previstas')
plt.ylabel('Classes Verdadeiras')

# Adicionar valores à matriz de confusão
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha='center', va='center',
                 color='white' if conf_matrix[i, j] > conf_matrix.max() / 2. else 'black')

plt.tight_layout()
plt.show()

# Relatório detalhado de classificação
report = classification_report(test.y, predicted_classes, target_names=CLASSES)
print(report)

# Identificar os piores casos (classes mais confundidas)
worse_cases = np.unravel_index(np.argmax(conf_matrix - np.eye(len(CLASSES)) * conf_matrix.max()), conf_matrix.shape)
print(f"Classes mais confundidas: Verdadeira - {CLASSES[worse_cases[0]]}, Prevista - {CLASSES[worse_cases[1]]}")
