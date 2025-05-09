# train.py
# Script to train a DQN agent on random 10×10 grids with obstacles, then plot performance

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ─── Environment Setup ────────────────────────────────────────────────────────
GRID_SIZE = 10
START     = (0, 0)
GOAL      = (GRID_SIZE - 1, GRID_SIZE - 1)

# Actions: up, down, left, right
ACTIONS    = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ─── Obstacle Sampling ─────────────────────────────────────────────────────────
NUM_WALLS = 15  # number of random obstacles per episode

def sample_obstacles(num_walls: int):
    """Uniformly sample `num_walls` obstacle cells, excluding start & goal."""
    cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    cells.remove(START)
    cells.remove(GOAL)
    return set(random.sample(cells, num_walls))

# ─── Validity Check ───────────────────────────────────────────────────────────
def is_valid(pos, obstacles):
    r, c = pos
    return (0 <= r < GRID_SIZE and
            0 <= c < GRID_SIZE and
            pos not in obstacles)

# ─── State Encoding ──────────────────────────────────────────────────────────
def state_to_tensor(state, goal, obstacles):
    """
    Build a 3×GRID×GRID tensor:
      - Channel 0: agent position (1.0)
      - Channel 1: goal position  (1.0)
      - Channel 2: obstacle mask  (1.0 where wall)
    """
    agent  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    goal_m = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    obs_m  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    agent[state] = 1.0
    goal_m[goal] = 1.0
    for (r, c) in obstacles:
        obs_m[r, c] = 1.0

    stacked = np.stack([agent, goal_m, obs_m], axis=0)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)  # shape (1,3,GRID,GRID)

# ─── DQN Network (CNN) ────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # -> (16×10×10)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # -> (32×10×10)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                                # 32*10*10 = 3200
            nn.Linear(32 * GRID_SIZE * GRID_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS))                 # 4 Q‑values
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EPISODES           = 2000
MAX_STEPS          = 100
GAMMA              = 0.99
LR                 = 1e-3
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY          = 0.995
MEMORY_SIZE        = 5000
BATCH_SIZE         = 64
TARGET_UPDATE_FREQ = 200

# ─── Setup Models & Optimizer ─────────────────────────────────────────────────
device      = torch.device("cpu")
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn   = nn.MSELoss()
memory    = deque(maxlen=MEMORY_SIZE)

epsilon            = EPS_START
step_ctr           = 0
reward_per_episode = []

# ─── Training Loop ──────────────────────────────────────────────────────────
for ep in range(1, EPISODES + 1):
    obstacles = sample_obstacles(NUM_WALLS)
    state     = START
    total_r   = 0

    for _ in range(MAX_STEPS):
        st_t = state_to_tensor(state, GOAL, obstacles).to(device)

        # ε-greedy action selection
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            with torch.no_grad():
                qs = policy_net(st_t)
            action = ACTIONS[torch.argmax(qs).item()]

        # Step environment
        next_state = (state[0] + action[0], state[1] + action[1])
        if not is_valid(next_state, obstacles):
            reward = -10
            next_state = state
        elif next_state == GOAL:
            reward = 20
        else:
            reward = -1

        done    = (next_state == GOAL)
        total_r += reward

        # Store transition
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Learning step
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions_, rewards, next_states, dones = zip(*batch)

            s_batch  = torch.cat([state_to_tensor(s, GOAL, obstacles) for s in states]).to(device)
            ns_batch = torch.cat([state_to_tensor(ns, GOAL, obstacles) for ns in next_states]).to(device)

            q_vals = policy_net(s_batch)
            q_next = target_net(ns_batch).detach()
            targets = q_vals.clone()

            for i in range(BATCH_SIZE):
                a_idx       = ACTION_IDX[actions_[i]]
                best_next_q = torch.max(q_next[i]).item()
                tgt_val     = rewards[i] + GAMMA * best_next_q * (0.0 if dones[i] else 1.0)
                targets[i, a_idx] = tgt_val

            loss = loss_fn(q_vals, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network periodically
        step_ctr += 1
        if step_ctr % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    # Decay epsilon and record reward
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    reward_per_episode.append(total_r)

    if ep % 100 == 0:
        print(f"Episode {ep}/{EPISODES}  Reward: {total_r:.1f}  ε: {epsilon:.3f}")

# ─── Save the trained model ────────────────────────────────────────────────────
torch.save(policy_net.state_dict(), "dqn_model.pth")
print("Training complete — model saved to dqn_model.pth")

# ─── Plot Training Performance ────────────────────────────────────────────────
episodes = list(range(1, EPISODES + 1))

plt.figure(figsize=(10, 5))
plt.plot(episodes, reward_per_episode, label='Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training: Reward per Episode')
plt.grid(True)
plt.legend()

# Moving average (window = 100)
window = 100
if EPISODES >= window:
    mov_avg = np.convolve(reward_per_episode, np.ones(window)/window, mode='valid')
    plt.plot(range(window, EPISODES + 1), mov_avg, label=f'{window}-Episode Moving Avg')
    plt.legend()

plt.show()
