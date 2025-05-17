# train.py
# Dueling Double-DQN with BFS-shaping, demonstration-augmented learning,
# and detailed logging — corrected replay sampling guard.

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

import gym
from gym import spaces
from gym.vector import AsyncVectorEnv

# ─── Hyperparameters & constants ──────────────────────────────────────────────
GRID_SIZE     = 10
START         = (0, 0)
GOAL          = (9, 9)
ACTIONS       = [(-1,0),(1,0),(0,-1),(0,1)]
NUM_WALLS     = 15
N_ENVS        = 4

EPISODES      = 5000
MAX_STEPS     = 100
GAMMA         = 0.99
LR            = 1e-3

EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.9995
REPLAY_SIZE   = 5000
BATCH_SIZE    = 64
TARGET_UPDATE = 200

DEMO_PATH     = "demos.pt"
DEMO_RATIO    = 0.25   # 25% of each batch from demos
BC_WEIGHT     = 1.0
MARGIN        = 0.8
MARGIN_WEIGHT = 1.0

# ─── Utility: BFS distance map for shaping ────────────────────────────────────
def compute_dist_map(obstacles):
    D = np.full((GRID_SIZE, GRID_SIZE), np.inf, np.float32)
    from collections import deque
    dq = deque([GOAL])
    D[GOAL] = 0
    while dq:
        r, c = dq.popleft()
        for dy, dx in ACTIONS:
            nr, nc = r + dy, c + dx
            if (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                and (nr, nc) not in obstacles
                and D[nr, nc] == np.inf):
                D[nr, nc] = D[r, c] + 1
                dq.append((nr, nc))
    return D

# ─── 1) Environment with solvability check & BFS shaping ─────────────────────
class GridEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, seed=None):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, (3, GRID_SIZE, GRID_SIZE), np.float32)
        self.action_space      = spaces.Discrete(len(ACTIONS))
        if seed is not None:
            random.seed(seed)

    def reset(self):
        # ensure solvable maze
        while True:
            cells = [(r,c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
            cells.remove(START); cells.remove(GOAL)
            self.obstacles = set(random.sample(cells, NUM_WALLS))
            self.dist_map  = compute_dist_map(self.obstacles)
            if np.isfinite(self.dist_map[START]):
                break
        self.state = START
        return self._get_obs(), {}

    def step(self, action):
        y, x = self.state
        dy, dx = ACTIONS[action]
        nxt = (y + dy, x + dx)

        if not (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE) or nxt in self.obstacles:
            base_r = -10
            nxt    = self.state
        elif nxt == GOAL:
            base_r = 20
        else:
            base_r = -1

        # shaping reward
        φ_s  = -self.dist_map[y, x]
        φ_s2 = -self.dist_map[nxt]
        shaping = GAMMA * φ_s2 - φ_s

        reward = base_r + shaping
        done   = (nxt == GOAL)
        self.state = nxt
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        agent = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
        goal  = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
        obs_m = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
        agent[self.state] = 1.0
        goal[GOAL]        = 1.0
        for (r,c) in self.obstacles:
            obs_m[r,c] = 1.0
        return np.stack([agent, goal, obs_m], axis=0)

# ─── 2) Dueling Double-DQN Network ─────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
        )
        self.adv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*GRID_SIZE*GRID_SIZE,128), nn.ReLU(),
            nn.Linear(128,len(ACTIONS)),
        )
        self.val = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*GRID_SIZE*GRID_SIZE,128), nn.ReLU(),
            nn.Linear(128,1),
        )

    def forward(self, x):
        x   = self.conv(x)
        adv = self.adv(x)
        val = self.val(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

# ─── 3) Load demonstrations ────────────────────────────────────────────────────
def load_demos():
    # We trust our own demos.pt, so allow full pickle load
    return torch.load(DEMO_PATH, weights_only=False)

# ─── 4) Main training loop with logging & corrected sampling ──────────────────
def main():
    torch.set_num_threads(torch.get_num_threads())
    torch.set_num_interop_threads(torch.get_num_threads())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    demos = load_demos()
    print(f"Loaded {len(demos)} demonstration transitions")

    # prepare vectorized environments
    venv = AsyncVectorEnv([lambda seed=i: GridEnv(seed=i) for i in range(N_ENVS)])

    policy_net = DuelingDQN().to(device)
    target_net = DuelingDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay    = deque(maxlen=REPLAY_SIZE)

    epsilon    = EPS_START
    step_count = 0
    rewards_hist = []

    print("🚀 Starting training...")
    for ep in trange(1, EPISODES+1, desc="Episodes"):
        obs, _     = venv.reset()
        ep_rewards = np.zeros(N_ENVS)
        done_mask  = np.zeros(N_ENVS, bool)

        total_rl_loss     = 0.0
        total_bc_loss     = 0.0
        total_margin_loss = 0.0
        num_updates       = 0
        ep_length         = 0

        for t in range(MAX_STEPS):
            ep_length = t+1
            st_t = torch.from_numpy(obs).to(device)
            with torch.no_grad():
                q_vals = policy_net(st_t)

            actions = []
            for i in range(N_ENVS):
                if random.random() < epsilon:
                    actions.append(random.randrange(len(ACTIONS)))
                else:
                    actions.append(torch.argmax(q_vals[i]).item())

            next_obs, rews, dones, truncs, _ = venv.step(actions)
            ep_rewards += rews
            done_mask  |= (dones | truncs)

            for i in range(N_ENVS):
                replay.append((obs[i], actions[i], rews[i], next_obs[i], dones[i]))
            obs = next_obs

            # ─── Correction made here ─────────────────────────
            # Determine how many from demos vs RL
            demo_n = int(BATCH_SIZE * DEMO_RATIO)
            rl_n   = BATCH_SIZE - demo_n
            # Only update when replay has at least rl_n samples
            if len(replay) >= rl_n:
                # sample RL transitions
                rl_batch   = random.sample(replay, rl_n)
                demo_batch = random.sample(demos, demo_n)
                # … rest of loss computation and optimizer.step() …
                # (omitted here for brevity; same as before)
                num_updates += 1
            # ────────────────────────────────────────────────────

            step_count += 1
            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done_mask.all():
                break

        # … logging and plotting as before … 

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
