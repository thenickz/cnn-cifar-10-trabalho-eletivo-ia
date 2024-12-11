import nbformat as nbf

# Criação do conteúdo do notebook
notebook = nbf.v4.new_notebook()

# Células do notebook
notebook.cells = [
    nbf.v4.new_markdown_cell("# CIFAR-10 Classification with CNN\nThis notebook demonstrates how to train a convolutional neural network (CNN) on the CIFAR-10 dataset using TensorFlow/Keras."),
    nbf.v4.new_code_cell("""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from utils import Dataset  # Ensure this contains the Dataset class
    """),
    nbf.v4.new_markdown_cell("## Class Names"),
    nbf.v4.new_code_cell("""
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """),
    nbf.v4.new_markdown_cell("## Load and Preprocess Data"),
    nbf.v4.new_code_cell("""
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
train = Dataset(x_train, y_train)
test = Dataset(x_test, y_test)

print(f"Train shape: {train.x.shape}, {train.y.shape}")
print(f"Test shape: {test.x.shape}, {test.y.shape}")

# Normalize data
train.x = train.x.astype('float32') / 255.0
test.x = test.x.astype('float32') / 255.0
    """),
    nbf.v4.new_markdown_cell("## Define the CNN Model"),
    nbf.v4.new_code_cell("""
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    """),
    nbf.v4.new_markdown_cell("## Train the Model"),
    nbf.v4.new_code_cell("""
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

history = cnn.fit(train.x, train.y,
                  validation_data=(test.x, test.y),
                  epochs=50,
                  batch_size=32,
                  callbacks=[early_stop, model_checkpoint])
    """),
    nbf.v4.new_markdown_cell("## Evaluate the Model"),
    nbf.v4.new_code_cell("""
test_loss, test_acc = cnn.evaluate(test.x, test.y)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")
    """),
    nbf.v4.new_markdown_cell("## Visualize Training Results"),
    nbf.v4.new_code_cell("""
# Plot loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
    """),
    nbf.v4.new_markdown_cell("## Make Predictions"),
    nbf.v4.new_code_cell("""
predictions = cnn.predict(test.x)
predicted_classes = np.argmax(predictions, axis=1)

# Display first 5 predictions
for i in range(5):
    print(f"Image {i+1}: True Class = {CLASSES[test.y[i][0]]}, Predicted Class = {CLASSES[predicted_classes[i]]}")
    """)
]

# Salvar o notebook em um arquivo
file_path = "CIFAR10_CNN_Notebook.ipynb"
with open(file_path, 'w', encoding='utf-8') as f:
    nbf.write(notebook, f)

file_path
