import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque

# --- 1. LSTM-DQN Neural Network ---
class LSTMDQN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4):
        super(LSTMDQN, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # batch_first=True -> input shape: (batch, seq_len, feature)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        out = self.relu(self.fc1(x))
        out, hidden = self.lstm(out, hidden)
        out = out[:, -1, :] # Take the last sequence output
        q_values = self.fc2(out)
        return q_values, hidden

# --- 2. Multi-Objective Replay Buffer ---
class MultiObjectiveReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, r1, r2, r3, next_state, done):
        self.buffer.append((state, action, r1, r2, r3, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, r1, r2, r3, next_state, done = map(np.stack, zip(*batch))
        return state, action, r1, r2, r3, next_state, done

    def __len__(self):
        return len(self.buffer)
    
    def load_from_csv(self, filepath):
        try:
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                state = np.array([row['state_N'], row['state_S'], row['state_E'], row['state_W']], dtype=np.float32)
                action = int(row['action'])
                r1, r2, r3 = float(row['r1_throughput']), float(row['r2_fairness']), float(row['r3_smoothness'])
                next_state = np.array([row['next_state_N'], row['next_state_S'], row['next_state_E'], row['next_state_W']], dtype=np.float32)
                done = bool(row['done'])
                self.push(state, action, r1, r2, r3, next_state, done)
        except Exception as e:
            print(f"Error loading CSV into buffer: {e}")

# --- 3. RL Agent ---
class LSTMDQNAgent:
    def __init__(self, state_dim=4, action_dim=4, lr=1e-3, gamma=0.99, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = LSTMDQN(state_dim, 64, action_dim).to(self.device)
        self.target_net = LSTMDQN(state_dim, 64, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = MultiObjectiveReplayBuffer()

    def select_action(self, state, epsilon, hidden=None):
        if random.random() < epsilon:
            return random.randrange(self.action_dim), hidden
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            q_values, next_hidden = self.policy_net(state_tensor, hidden)
            return q_values.argmax().item(), next_hidden

    def train_step(self, weights):
        if len(self.memory) < self.batch_size: return 0.0

        w1, w2, w3 = weights
        states, actions, r1s, r2s, r3s, next_states, dones = self.memory.sample(self.batch_size)

        # Apply Pareto Weights to Scalarize Reward (Eq 1 from paper)
        scalarized_rewards = (w1 * r1s) + (w2 * r2s) + (w3 * r3s)

        states = torch.FloatTensor(states).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(scalarized_rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values, _ = self.policy_net(states)
        current_q = q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())