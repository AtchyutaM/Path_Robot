# findpath.py
# Inference module for the trained Dueling Double‑DQN agent

import torch
import numpy as np
import torch.nn as nn

# ─── Constants ────────────────────────────────────────────────────────────────
GRID_SIZE  = 10
MODEL_PATH = "dqn_model.pth"
ACTIONS    = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_STEPS  = 100

# ─── Dueling Double‑DQN Network (must match train.py) ─────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.adv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * GRID_SIZE * GRID_SIZE, 128), nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )
        self.val = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * GRID_SIZE * GRID_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x   = self.conv(x)
        adv = self.adv(x)
        val = self.val(x)
        # Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]
        return val + adv - adv.mean(dim=1, keepdim=True)


# ─── Utilities ────────────────────────────────────────────────────────────────
def state_to_tensor(state, goal, obstacles: set[tuple[int,int]]):
    """
    Build a 3×GRID×GRID tensor:
      - Channel 0: agent location (1.0)
      - Channel 1: goal location  (1.0)
      - Channel 2: obstacle mask  (1.0 where wall)
    Returns a torch.FloatTensor shaped (1, 3, GRID, GRID).
    """
    agent  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    goal_m = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    obs_m  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    agent[state]  = 1.0
    goal_m[goal]  = 1.0
    for (r, c) in obstacles:
        obs_m[r, c] = 1.0

    stacked = np.stack([agent, goal_m, obs_m], axis=0)
    return torch.from_numpy(stacked).unsqueeze(0)  # shape: (1,3,GRID,GRID)

def is_valid(pos, obstacles: set[tuple[int,int]]):
    """Return True if pos is inside the grid and not in obstacles."""
    r, c = pos
    return (0 <= r < GRID_SIZE) and (0 <= c < GRID_SIZE) and (pos not in obstacles)


# ─── Path‑finding Function ────────────────────────────────────────────────────
def find_path(obstacles: set[tuple[int,int]],
              start:     tuple[int,int],
              goal:      tuple[int,int]) -> list[tuple[int,int]]:
    """
    Uses the trained Dueling Double‑DQN to perform a greedy rollout.
    Returns a list of (y, x) coordinates from start toward goal.
    """
    # 1) Load model onto appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DuelingDQN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    path  = [start]
    state = start

    for _ in range(MAX_STEPS):
        # Encode current state
        st_t = state_to_tensor(state, goal, obstacles).to(device)

        # Forward pass to get Q-values
        with torch.no_grad():
            q_vals = model(st_t)         # shape: (1,4)
        best_idx = int(q_vals.argmax(dim=1).item())
        dy, dx   = ACTIONS[best_idx]
        nxt      = (state[0] + dy, state[1] + dx)

        # Stop if move is invalid or would loop
        if not is_valid(nxt, obstacles) or nxt in path:
            break

        path.append(nxt)
        if nxt == goal:
            break
        state = nxt

    return path
