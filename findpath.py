# findpath.py
# RL inference module: loads a trained DQN model and computes a path given obstacles

import torch
import numpy as np
import torch.nn as nn

# ─── Network Definition ───────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

# ─── Constants & Utilities ──────────────────────────────────────────────────
GRID_SIZE = 10
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
MAX_STEPS = 100
MODEL_PATH = "dqn_model.pth"  # path to the saved model file

def state_to_tensor(state, goal):
    """
    Encode current state and goal into a flattened tensor.
    - state, goal: (y, x) tuples
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    grid[state] = 1.0
    grid[goal] = 0.5
    return torch.tensor(grid.flatten(), dtype=torch.float32).unsqueeze(0)

def is_valid(pos, obstacles):
    """
    Check if a position is within bounds and not in the obstacles set.
    """
    r, c = pos
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and pos not in obstacles

# ─── Path-Finding Function ─────────────────────────────────────────────────
def find_path(obstacles: set[tuple[int,int]],
              start: tuple[int,int],
              goal: tuple[int,int]) -> list[tuple[int,int]]:
    """
    Load the trained DQN model and roll out a greedy path from start to goal.
    - obstacles: set of (y, x) tuples
    - start, goal: (y, x)
    Returns a list of (y, x) coordinates.
    """
    # 1. Load model
    model = DQN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 2. Greedy rollout
    path = [start]
    state = start
    for _ in range(MAX_STEPS):
        # a) encode state
        state_tensor = state_to_tensor(state, goal)
        # b) predict Q-values
        with torch.no_grad():
            q_vals = model(state_tensor)
        # c) pick best action
        best_action = ACTIONS[torch.argmax(q_vals).item()]
        # d) compute next state
        next_state = (state[0] + best_action[0], state[1] + best_action[1])

        # e) stop if invalid or looping
        if not is_valid(next_state, obstacles) or next_state in path:
            break

        path.append(next_state)
        if next_state == goal:
            break
        state = next_state

    return path
