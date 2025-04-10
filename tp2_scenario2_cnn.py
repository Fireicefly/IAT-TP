import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import kagglehub

# Configuration
BATCH_SIZE = 32
EPOCHS = 15
TARGET_SIZE = (64, 64)
INPUT_SHAPE = (64, 64, 3)
KERNEL_SIZE = (3, 3)
NUM_CLASSES = 2  # Véhicule (1) ou non-véhicule (0)
LEARNING_RATE = 0.001

# Télécharger le dataset
path = kagglehub.dataset_download("brsdincer/vehicle-detection-image-set")
print("Path to dataset files:", path)

# Fonction pour charger et prétraiter les images
def load_dataset(folder_path, label):
    dataset = []
    images = []
    labels = []
    
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir en RGB
            img = cv2.resize(img, TARGET_SIZE)
            # Normaliser les données
            img = img / 255.0
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Charger le dataset
print("Chargement des images de véhicules...")
vehicle_images, vehicle_labels = load_dataset(path + "/data/vehicles", label=1)
print(f"Chargé {len(vehicle_images)} images de véhicules")

print("Chargement des images de non-véhicules...")
non_vehicle_images, non_vehicle_labels = load_dataset(path + "/data/non-vehicles", label=0)
print(f"Chargé {len(non_vehicle_images)} images de non-véhicules")

# Combiner les datasets
X = np.concatenate((vehicle_images, non_vehicle_images), axis=0)
y = np.concatenate((vehicle_labels, non_vehicle_labels), axis=0)

# Diviser en train, validation et test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Taille du dataset d'entraînement: {len(X_train)}")
print(f"Taille du dataset de validation: {len(X_val)}")
print(f"Taille du dataset de test: {len(X_test)}")

# Définition du modèle CNN
def create_cnn_model():
    model = Sequential([
        # Première couche de convolution
        Conv2D(32, KERNEL_SIZE, activation='relu', padding='same', input_shape=INPUT_SHAPE),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Deuxième couche de convolution
        Conv2D(64, KERNEL_SIZE, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Troisième couche de convolution
        Conv2D(128, KERNEL_SIZE, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Aplatir les caractéristiques
        Flatten(),
        
        # Couches fully connected
        Dense(256, activation='relu'),
        Dropout(0.5),  # Pour éviter le sur-apprentissage
        Dense(NUM_CLASSES, activation='softmax')  # Sortie pour classification binaire
    ])
    
    # Compiler le modèle
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Création d'une classe de callback pour visualiser l'entraînement en temps réel
class LivePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LivePlotCallback, self).__init__()
        self.epochs = []
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        
        # Configuration des plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.suptitle('Progression de l\'entraînement')
        
        # Plot pour l'accuracy
        self.acc_lines, = self.ax1.plot([], [], 'b-', label='Accuracy (entraînement)')
        self.val_acc_lines, = self.ax1.plot([], [], 'r-', label='Accuracy (validation)')
        self.ax1.set_title('Accuracy du modèle')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy')
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Plot pour la loss
        self.loss_lines, = self.ax2.plot([], [], 'b-', label='Loss (entraînement)')
        self.val_loss_lines, = self.ax2.plot([], [], 'r-', label='Loss (validation)')
        self.ax2.set_title('Loss du modèle')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True)
        self.ax2.legend()
        
        plt.tight_layout()
        plt.ion()  # Mode interactif
        plt.show(block=False)
        
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        
        # Mise à jour des données
        self.acc_lines.set_data(self.epochs, self.acc)
        self.val_acc_lines.set_data(self.epochs, self.val_acc)
        self.loss_lines.set_data(self.epochs, self.loss)
        self.val_loss_lines.set_data(self.epochs, self.val_loss)
        
        # Ajuster les limites des axes
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Mettre à jour le titre
        self.fig.suptitle(f'Progression de l\'entraînement (Epoch {epoch+1}/{EPOCHS})')
        
        # Actualiser les plots
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Afficher les métriques actuelles
        print(f"Epoch {epoch+1}/{EPOCHS} terminée")
        print(f"Accuracy (entraînement): {logs.get('accuracy'):.4f}, Accuracy (validation): {logs.get('val_accuracy'):.4f}")
        print(f"Loss (entraînement): {logs.get('loss'):.4f}, Loss (validation): {logs.get('val_loss'):.4f}")

# Créer le modèle
model = create_cnn_model()

# Résumé du modèle
model.summary()

# Callbacks
live_plot = LivePlotCallback()
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Entraînement du modèle
print("Début de l'entraînement...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[live_plot, checkpoint, early_stopping]
)

# Désactiver le mode interactif
plt.ioff()

# Évaluation sur le dataset de test
print("Évaluation sur le dataset de test...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Prédictions sur le dataset de test
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Non-véhicule', 'Véhicule'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Matrice de confusion')
plt.savefig('confusion_matrix.png')
plt.show()

# Rapport de classification
print("\nRapport de classification:")
print(classification_report(y_test, y_pred, target_names=['Non-véhicule', 'Véhicule']))

# Tracer les courbes d'accuracy et de loss finales
plt.figure(figsize=(15, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy (entraînement)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation)')
plt.title('Accuracy du modèle')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss (entraînement)')
plt.plot(history.history['val_loss'], label='Loss (validation)')
plt.title('Loss du modèle')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# Visualiser quelques prédictions
def plot_sample_predictions(X_test, y_test, y_pred, num_samples=5):
    plt.figure(figsize=(15, 10))
    indices = np.random.choice(range(len(X_test)), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_test[idx])
        true_label = "Véhicule" if y_test[idx] == 1 else "Non-véhicule"
        pred_label = "Véhicule" if y_pred[idx] == 1 else "Non-véhicule"
        color = "green" if y_test[idx] == y_pred[idx] else "red"
        plt.title(f"Vrai: {true_label}\nPréd: {pred_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

# Afficher quelques prédictions
plot_sample_predictions(X_test, y_test, y_pred, num_samples=5)

# Sauvegarde du modèle final
model.save('vehicle_detection_cnn.h5')
print("Modèle sauvegardé sous 'vehicle_detection_cnn.h5'")