# findpath.py
import torch
import numpy as np
import torch.nn as nn

# ─── Constants ────────────────────────────────────────────────────────────────
GRID_SIZE = 10
MODEL_PATH = "dqn_model.pth"   # trained weights from train.py
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
ACTION_IDX = {a:i for i,a in enumerate(ACTIONS)}
MAX_STEPS = 100

# ─── Network ──────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                    # 32 * GRID_SIZE * GRID_SIZE
            nn.Linear(32 * GRID_SIZE * GRID_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS))
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# ─── Utilities ────────────────────────────────────────────────────────────────
def state_to_tensor(state, goal, obstacles:set[tuple[int,int]]):
    # Build three GRID×GRID channels
    agent  = np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.float32)
    goal_m = np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.float32)
    obs_m  = np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.float32)

    agent [state ] = 1.0
    goal_m[goal  ] = 1.0
    for (r,c) in obstacles:
        obs_m[r,c] = 1.0

    stacked = np.stack([agent, goal_m, obs_m], axis=0)  # shape=(3,GRID,GRID)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)  # (1,3,GRID,GRID)

def is_valid(pos, obstacles):
    r,c = pos
    return 0<=r<GRID_SIZE and 0<=c<GRID_SIZE and pos not in obstacles

# ─── Path‑finding ─────────────────────────────────────────────────────────────
def find_path(obstacles:set[tuple[int,int]],
              start: tuple[int,int],
              goal:  tuple[int,int]) -> list[tuple[int,int]]:

    # 1) Load the trained model
    model = DQN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 2) Greedy rollout
    path = [start]
    state = start

    for _ in range(MAX_STEPS):
        # a) encode input
        st_t = state_to_tensor(state, goal, obstacles)
        # b) forward pass
        with torch.no_grad():
            q_vals = model(st_t)
        # c) pick best action
        best = ACTIONS[torch.argmax(q_vals).item()]
        # d) step
        nxt = (state[0]+best[0], state[1]+best[1])

        # e) stop if invalid or loop
        if not is_valid(nxt, obstacles) or nxt in path:
            break

        path.append(nxt)
        if nxt == goal:
            break
        state = nxt

    return path
