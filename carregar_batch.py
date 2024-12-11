import pickle
import numpy as np
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# Diretório do CIFAR-10
cifar10_dir = r"cifar-10-batches-py"

# Combinar os batches de treino
X_train = []
Y_train = []

for i in range(1, 6):  # Itera pelos 5 data_batches
    batch = unpickle(os.path.join(cifar10_dir, f"data_batch_{i}"))
    X_train.append(batch[b'data'])
    Y_train += batch[b'labels']

# Converter para arrays numpy
X_train = np.vstack(X_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
Y_train = np.array(Y_train)

# Carregar o batch de teste
test_batch = unpickle(os.path.join(cifar10_dir, "test_batch"))
X_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
Y_test = np.array(test_batch[b'labels'])

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
