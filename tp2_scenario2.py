import os
import cv2
import numpy as np
import random
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Importer matplotlib pour tracer les courbes
from sklearn.model_selection import train_test_split  # Importer train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Charger le dataset depuis Kaggle
import kagglehub

# Download latest version
path = kagglehub.dataset_download("brsdincer/vehicle-detection-image-set")
print("Path to dataset files:", path)

# Extraction des descripteurs HOG
def extract_hog_features(image): 
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, visualize=True, block_norm='L2-Hys') 
    return features

def load_dataset(folder_path, label):
    dataset = []
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Convertir en niveaux de gris
        img = cv2.resize(img, (64, 64)) # Redimensionner
        features = extract_hog_features(img) # Extraire les caractéristiques HOG
        dataset.append({'features': features, 'label': label})
    return dataset

# Update the paths to point to the extracted dataset directories
vehicle_images = load_dataset(path + "/data/vehicles", label=1)
non_vehicle_images = load_dataset(path + "/data/non-vehicles", label=0)
full_dataset = vehicle_images + non_vehicle_images

# Diviser le dataset en train (70%) et test (30%)
train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.3, random_state=42)

# Normaliser les caractéristiques pour le dataset d'entraînement
scaler = StandardScaler()
features_matrix = np.array([data['features'] for data in train_dataset])
scaled_features = scaler.fit_transform(features_matrix)

# Mettre à jour le dataset d'entraînement avec les caractéristiques normalisées
for i, data in enumerate(train_dataset):
    train_dataset[i]['features'] = tuple(scaled_features[i])

# Normaliser les caractéristiques pour le dataset de test
test_features_matrix = np.array([data['features'] for data in test_dataset])
scaled_test_features = scaler.transform(test_features_matrix)

# Mettre à jour le dataset de test avec les caractéristiques normalisées
for i, data in enumerate(test_dataset):
    test_dataset[i]['features'] = tuple(scaled_test_features[i])

# Convertir en tuple pour l'utiliser comme clé dans la Q-Table
# Définition de l'environnement
class Environment:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.dataset[self.current_index]['features']
    
    def step(self, action):
        reward = 1 if action == self.dataset[self.current_index]['label'] else -1

        self.current_index += 1
        done = self.current_index >= len(self.dataset)

        next_state = self.dataset[self.current_index]['features'] if not done else None
        return next_state, reward, done
    
# Agent Q-Learning
class QLearningAgent:
    def __init__(self, action_space):
        self.q_table = {}
        self.action_space = action_space
        self.learning_rate = 0.001
        self.gamma = 0.7

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        return np.argmax(self.q_table[state]) if random.random() > 0.1 else random.randint(0, self.action_space - 1)
    
    def update_q_value(self, state, action, reward, next_state):
        if next_state is None:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table.get(next_state, np.zeros(self.action_space)))

        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

# Simulation
env = Environment(train_dataset)
agent = QLearningAgent(action_space=2) # Deux actions : 0 (Non-véhicule), 1 (Véhicule)

total_rewards_per_episode = []  # Liste pour stocker les récompenses totales par épisode

train_accuracies = []  # Liste pour stocker les accuracy sur le dataset d'entraînement
test_accuracies = []   # Liste pour stocker les accuracy sur le dataset de test

# Afficher la Q-Table
# print("Q-Table (partielle):", list(agent.q_table.items())[:5])

def evaluate_agent(agent, dataset):
    correct_predictions = 0
    total_predictions = len(dataset)
    true_positives = 0
    false_negatives = 0
    total_loss = 0

    y_true = []  # Liste pour les vraies étiquettes
    y_pred = []  # Liste pour les prédictions

    for data in dataset:
        state = data['features']
        true_label = data['label']
        predicted_action = agent.choose_action(state)

        # Ajouter les étiquettes pour la matrice de confusion
        y_true.append(true_label)
        y_pred.append(predicted_action)

        # Accuracy: Count correct predictions
        if predicted_action == true_label:
            correct_predictions += 1

        # Recall: Count true positives and false negatives
        if true_label == 1:  # Positive class (e.g., "Vehicle")
            if predicted_action == 1:
                true_positives += 1
            else:
                false_negatives += 1

        # Loss: Use a simple loss function (e.g., 0 for correct, 1 for incorrect)
        loss = 1 if predicted_action != true_label else 0
        total_loss += loss

    # Calculate metrics
    accuracy = correct_predictions / total_predictions
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    average_loss = total_loss / total_predictions

    return accuracy, average_loss, recall, y_true, y_pred

# Evaluate the agent after training
accuracy, loss, recall, y_true, y_pred = evaluate_agent(agent, test_dataset)

for episode in range(10):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    total_rewards_per_episode.append(total_reward)  # Enregistrer la récompense totale
    print(f"Épisode {episode + 1} terminé avec une récompense totale de {total_reward}")

    # Évaluer l'agent sur le dataset d'entraînement
    train_accuracy, _, _, _, _ = evaluate_agent(agent, train_dataset)
    train_accuracies.append(train_accuracy)
    print(f"Épisode {episode + 1} - Train Accuracy: {train_accuracy:.2f}")

    # Évaluer l'agent sur le dataset de test
    test_accuracy, _, _, _, _ = evaluate_agent(agent, test_dataset)
    test_accuracies.append(test_accuracy)

print(f"Accuracy: {accuracy:.2f}")
print(f"Loss: {loss:.2f}")
print(f"Recall: {recall:.2f}")

# Tracer la courbe des récompenses totales
plt.plot(range(1, len(total_rewards_per_episode) + 1), total_rewards_per_episode, marker='o')
plt.title("Évolution de la récompense totale par épisode")
plt.xlabel("Épisode")
plt.ylabel("Récompense totale")
plt.grid()
plt.show()

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_true, y_pred)

# Tracer les courbes des accuracy
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", marker='o')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy", marker='x')
plt.title("Accuracy en fonction des épisodes")
plt.xlabel("Épisode")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()