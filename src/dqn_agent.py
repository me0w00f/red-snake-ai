import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        self.var = M2 / (self.count + batch_count)
        self.count += batch_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # 添加设备信息打印

        self.model = DQN(state_size, 64, action_size).to(self.device)
        self.target_model = DQN(state_size, 64, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # 添加状态标准化
        self.state_normalizer = RunningMeanStd(shape=state_size)
        
        # 添加训练统计
        self.episode_rewards = []
        self.max_rewards = deque(maxlen=100)
        self.avg_q_values = deque(maxlen=100)
        self.max_q_values = deque(maxlen=100)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def normalize_state(self, state):
        try:
            if isinstance(state, np.ndarray):
                if state.size > 0:  # 确保数组不为空
                    self.state_normalizer.update(state.reshape(1, -1))
                    return self.state_normalizer.normalize(state)
            return state
        except Exception as e:
            print(f"State normalization error: {e}")
            return state

    def act(self, state):
        state = self.normalize_state(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_values = self.model(state)
            self.avg_q_values.append(action_values.mean().item())
            self.max_q_values.append(action_values.max().item())
            
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return {
                'loss': 0.0,
                'td_error': 0.0,
                'q_value_mean': 0.0,
                'q_value_max': 0.0
            }
        
        # 批量处理数据以提高GPU利用率
        minibatch = random.sample(self.memory, batch_size)
        
        # 首先将数据转换为numpy数组
        states = np.array([self.normalize_state(s[0]) for s in minibatch])
        next_states = np.array([self.normalize_state(s[3]) for s in minibatch])
        
        # 然后一次性转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor([s[1] for s in minibatch]).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in minibatch]).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor([s[4] for s in minibatch]).to(self.device)

        # 批量计算当前Q值
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        # 批量计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # 计算损失并更新
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 收集统计数据
        td_errors = abs(target_q_values - current_q_values.squeeze()).detach()
        q_values = current_q_values.detach()

        stats = {
            'loss': loss.item(),
            'td_error': td_errors.mean().item(),
            'q_value_mean': q_values.mean().item(),
            'q_value_max': q_values.max().item()
        }

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return stats

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_normalizer': {
                'mean': self.state_normalizer.mean,
                'var': self.state_normalizer.var,
                'count': self.state_normalizer.count
            }
        }
        torch.save(save_dict, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加载标准化参数
        self.state_normalizer.mean = checkpoint['state_normalizer']['mean']
        self.state_normalizer.var = checkpoint['state_normalizer']['var']
        self.state_normalizer.count = checkpoint['state_normalizer']['count']
