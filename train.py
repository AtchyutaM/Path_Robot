# train.py
# Script to train a DQN agent on a grid navigation task and save the trained model

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ---- Grid setup ----
GRID_SIZE = 10
START = (0, 0)
GOAL = (9, 9)
OBSTACLES = {
    (1, 2), (1, 3), (2, 3), (3, 3),
    (4, 1), (4, 2), (4, 3), (5, 5),
    (6, 5), (7, 5), (7, 6), (7, 7)
}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ---- DQN Model ----
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(GRID_SIZE * GRID_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(ACTIONS))
        )

    def forward(self, x):
        return self.net(x)

# ---- Utility Functions ----
def is_valid(pos):
    r, c = pos
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and pos not in OBSTACLES


def state_to_tensor(state, goal):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    grid[state] = 1.0
    grid[goal] = 0.5
    return torch.tensor(grid.flatten(), dtype=torch.float32).unsqueeze(0)


def choose_action(state_tensor, epsilon, model):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    with torch.no_grad():
        q_vals = model(state_tensor)
    return ACTIONS[torch.argmax(q_vals).item()]

# ---- Hyperparameters ----
gamma = 0.99
alpha = 0.001
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995
EPISODES = 1000
MAX_STEPS = 100
TARGET_UPDATE_FREQ = 100
BATCH_SIZE = 32
MEMORY_SIZE = 5000

# ---- Initialization ----n
device = torch.device("cpu")
model = DQN().to(device)
target_model = DQN().to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()
memory = deque(maxlen=MEMORY_SIZE)

epsilon = epsilon_start
reward_per_episode = []
step_counter = 0

# ---- Training Loop ----
for episode in range(EPISODES):
    state = START
    total_reward = 0
    for step in range(MAX_STEPS):
        state_tensor = state_to_tensor(state, GOAL).to(device)
        action = choose_action(state_tensor, epsilon, model)
        next_state = (state[0] + action[0], state[1] + action[1])

        # Compute reward and validity
        if not is_valid(next_state):
            next_state = state
            reward = -10
        elif next_state == GOAL:
            reward = 20
        else:
            reward = -1

        done = (next_state == GOAL)
        total_reward += reward

        # Store transition
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Sample and update
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            state_tensors = torch.cat([state_to_tensor(s, GOAL) for s in states]).to(device)
            next_state_tensors = torch.cat([state_to_tensor(ns, GOAL) for ns in next_states]).to(device)

            q_values = model(state_tensors)
            next_q_values = target_model(next_state_tensors)

            targets = q_values.clone().detach()
            for i in range(BATCH_SIZE):
                a_idx = ACTION_IDX[actions[i]]
                max_next_q = torch.max(next_q_values[i]).item()
                target_val = rewards[i] + gamma * max_next_q * (0 if dones[i] else 1)
                targets[i][a_idx] = target_val

            loss = loss_fn(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network periodically
        step_counter += 1
        if step_counter % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(model.state_dict())

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    reward_per_episode.append(total_reward)

# ---- Save the trained model ----
torch.save(model.state_dict(), 'dqn_model.pth')
print('Training complete. Model saved to dqn_model.pth')
