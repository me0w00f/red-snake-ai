import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.memory = deque(maxlen=50000)  # Larger replay buffer
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # Slower decay for better exploration
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_model = DQN(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        losses = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            target_f = self.model(state)
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return np.mean(losses)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_size': self.hidden_size,
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        # Support both old format (raw state_dict) and new format (dict with metadata)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
        else:
            self.model.load_state_dict(checkpoint)
        self.target_model.load_state_dict(self.model.state_dict())
