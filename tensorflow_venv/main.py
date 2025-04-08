import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub

# Hyperparamètres
BATCH_SIZE = 128
TARGET_SIZE = (64, 64)
EPOCHS = 16
NUM_CLASSES = 10
LEARNING_RATE = 0.0001
INPUT_SHAPE = (64, 64, 3)

# Désactiver CUDA pour éviter les erreurs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

# Modèle CNN
model = Sequential([
    tf.keras.layers.Input(shape=INPUT_SHAPE),  
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax') 
])

# Compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Évaluation
loss, accuracy = model.evaluate(test_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')
