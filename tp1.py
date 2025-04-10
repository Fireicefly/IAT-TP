import suppress_logs
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import time

# Enable interactive mode for matplotlib
plt.ion()

# Custom callback for live plotting with a single updating window
class LivePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LivePlotCallback, self).__init__()
        self.epochs = []
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        
        # Set up the plot once
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.suptitle('Training Progress')
        
        # Setup for accuracy plot
        self.acc_lines, = self.ax1.plot([], [], 'b-', label='Training Accuracy')
        self.val_acc_lines, = self.ax1.plot([], [], 'r-', label='Validation Accuracy')
        self.ax1.set_title('Model Accuracy')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy')
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Setup for loss plot
        self.loss_lines, = self.ax2.plot([], [], 'b-', label='Training Loss')
        self.val_loss_lines, = self.ax2.plot([], [], 'r-', label='Validation Loss')
        self.ax2.set_title('Model Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True)
        self.ax2.legend()
        
        plt.tight_layout()
        plt.show(block=False)
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{EPOCHS} started...")
        
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        
        # Update the plot data
        self.acc_lines.set_data(self.epochs, self.acc)
        self.val_acc_lines.set_data(self.epochs, self.val_acc)
        self.loss_lines.set_data(self.epochs, self.loss)
        self.val_loss_lines.set_data(self.epochs, self.val_loss)
        
        # Adjust axis limits
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Update figure title
        self.fig.suptitle(f'Training Progress (Epoch {epoch+1}/{EPOCHS})')
        
        # Draw the updated plots
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Print current metrics
        print(f"Epoch {epoch+1}/{EPOCHS} completed")
        print(f"Training Accuracy: {logs.get('accuracy'):.4f}, Validation Accuracy: {logs.get('val_accuracy'):.4f}")
        print(f"Training Loss: {logs.get('loss'):.4f}, Validation Loss: {logs.get('val_loss'):.4f}")

# Hyperparamètres
BATCH_SIZE = 32
TARGET_SIZE = (64, 64)
EPOCHS = 16
NUM_CLASSES = 10
LEARNING_RATE = 0.0005
INPUT_SHAPE = (64, 64, 3)

# Télécharger le dataset
path = kagglehub.dataset_download("idrisskh/obstacles-dataset")

print(f'Dataset téléchargé dans {path}')

dataset_path = os.path.join(path, "obstacles dataset")
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "test")  # Répertoire de validation

# Utiliser 60% des données de validation pour les tests et 40% pour la validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.4)  # 40% pour les tests

# Générateur pour les données d'entraînement
train_generator = datagen.flow_from_directory(
    train_dir, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
)

# Générateur pour les données de validation (80% de validation, 20% de test)
val_generator = datagen.flow_from_directory(
    val_dir, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', subset='training'
)

# Générateur pour les données de test (20% du répertoire val_dir)
test_generator = datagen.flow_from_directory(
    val_dir, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', subset='validation'
)
KERNEL_SIZE = (3, 3)
# Modèle CNN
model = Sequential([
    tf.keras.layers.Input(shape=INPUT_SHAPE),  
    Conv2D(32, KERNEL_SIZE, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax') 
])

# Compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create the live plot callback
live_plot = LivePlotCallback()

# Entraînement avec visualisation en temps réel
print("Starting training...")
history = model.fit(
    train_generator, 
    validation_data=val_generator, 
    epochs=EPOCHS,
    callbacks=[live_plot]
)

# Turn off interactive mode after training
plt.ioff()

# Évaluation
print("Evaluating model on test data...")
loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Create a final summary figure (this will be a new window)
plt.figure(figsize=(15, 10))

# Plot final accuracy and loss curves
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()