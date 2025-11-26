import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

# --- 1. DEFINICIÓN DEL CEREBRO (RED NEURONAL) ---
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        # Capas Convolucionales (Para procesar la imagen)
        # Entrada: Imagen pequeña en escala de grises
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Capas Lineales (Para tomar decisiones)
        # Calculamos el tamaño de entrada dinámicamente
        self.fc_input_dim = self._get_conv_out(input_shape)
        
        self.layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions) # Salida: Q-Values para cada acción
        )

    def _get_conv_out(self, shape):
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.features(x)
        x = self.layers(x)
        return x

# --- 2. EL AGENTE (CONTROLADOR) ---
class DQNAgent:
    def __init__(self):
        # Hiperparámetros (Configuración de aprendizaje)
        self.state_shape = (1, 80, 80) # Imagen: 1 canal (gris), 80x80 px
        self.action_size = 3           # Acciones: [0:Recto, 1:Izq, 2:Der]
        self.gamma = 0.99              # Importancia del futuro (0.99 = muy previsor)
        self.epsilon = 1.0             # Exploración inicial (1.0 = 100% aleatorio al inicio)
        self.epsilon_min = 0.05        # Mínimo de exploración (5%)
        self.epsilon_decay = 0.995     # Qué tan rápido deja de explorar
        self.learning_rate = 0.001
        self.batch_size = 32           # Cuántos recuerdos aprende a la vez
        
        # Memoria (Buffer)
        self.memory = deque(maxlen=2000)
        
        # Dispositivo (Detectar si hay GPU, sino usar CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cerebro cargado en: {self.device}")

        # Crear Redes (Principal y Objetivo)
        self.model = DQN(self.state_shape, self.action_size).to(self.device)
        self.target_model = DQN(self.state_shape, self.action_size).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        """Sincroniza la red estable con la red de aprendizaje"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Guarda una experiencia en la memoria"""
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """Decide qué hacer basado en el estado actual"""
        # Exploración (Aleatorio)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Explotación (Usar el cerebro)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def train(self):
        """Entrena la red neuronal re-viviendo recuerdos pasados"""
        if len(self.memory) < self.batch_size:
            return 0 # No hay suficientes recuerdos aún
        
        # Tomar una muestra aleatoria de recuerdos
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Preparar datos para PyTorch
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)
        
        # Ecuación de Bellman (La magia del Q-Learning)
        # Q_nuevo = Recompensa + Gamma * max(Q_futuro)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Predicción actual
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Calcular error y corregir la red
        loss = self.loss_fn(current_q_values.squeeze(), targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Reducir la exploración poco a poco
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename, map_location=self.device))
            self.epsilon = 0.1 # Si cargamos un modelo, exploramos poco
            print("Modelo cargado exitosamente.")
        else:
            print("No se encontró modelo guardado.")