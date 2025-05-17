# train.py
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
ACTIONS       = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NUM_WALLS     = 15
N_ENVS        = 4

EPISODES      = 10000
MAX_STEPS     = 100
GAMMA         = 0.99
LR            = 1e-3

EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.9995
REPLAY_SIZE   = 5000
BATCH_SIZE    = 64
TARGET_UPDATE = 200

DEMO_PATH     = "demos.pt"
DEMO_RATIO    = 0.25    # fraction of each batch from demos
BC_WEIGHT     = 1.0
MARGIN        = 0.8
MARGIN_WEIGHT = 1.0

# ─── Utility: BFS distance map for potential‐based shaping ────────────────────
def compute_dist_map(obstacles):
    """
    Compute shortest‐path distance from every cell to GOAL via BFS,
    given a fixed set of obstacles.
    Unreachable cells remain at +inf.
    """
    D = np.full((GRID_SIZE, GRID_SIZE), np.inf, dtype=np.float32)
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

# ─── 1) Environment: solvable sampling + BFS shaping ──────────────────────────
class GridEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, seed=None):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, (3, GRID_SIZE, GRID_SIZE), np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))
        if seed is not None:
            random.seed(seed)

    def reset(self):
        # Repeatedly sample until the maze is solvable
        while True:
            cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
            cells.remove(START); cells.remove(GOAL)
            self.obstacles = set(random.sample(cells, NUM_WALLS))
            self.dist_map  = compute_dist_map(self.obstacles)
            if np.isfinite(self.dist_map[START]):
                break
        self.state = START
        return self._get_obs(), {}

    def step(self, action):
        # Apply action and compute base reward
        y, x = self.state
        dy, dx = ACTIONS[action]
        nxt = (y + dy, x + dx)

        if not (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE) or nxt in self.obstacles:
            base_r = -10
            nxt = self.state
        elif nxt == GOAL:
            base_r = 20
        else:
            base_r = -1

        # Potential‐based shaping: Φ(s) = –D[s]
        φ_s  = -self.dist_map[y, x]
        φ_s2 = -self.dist_map[nxt]
        shaping = GAMMA * φ_s2 - φ_s

        reward = base_r + shaping
        done   = (nxt == GOAL)
        self.state = nxt
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # 3‐channel tensor: [agent_map, goal_map, obstacle_map]
        agent = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
        goal  = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
        obs_m = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
        agent[self.state] = 1.0
        goal[GOAL]        = 1.0
        for (r,c) in self.obstacles:
            obs_m[r, c] = 1.0
        return np.stack([agent, goal, obs_m], axis=0)

# ─── 2) Dueling Double‐DQN Network ─────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16,32, 3, padding=1), nn.ReLU(),
        )
        self.adv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*GRID_SIZE*GRID_SIZE,128), nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )
        self.val = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*GRID_SIZE*GRID_SIZE,128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x   = self.conv(x)
        adv = self.adv(x)
        val = self.val(x)
        # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        return val + adv - adv.mean(dim=1, keepdim=True)

# ─── 3) Load demonstration transitions ─────────────────────────────────────────
def load_demos():
    # demos.pt contains list of (obs_numpy, action_int) pairs
    return torch.load(DEMO_PATH, weights_only=False)

# ─── 4) Training loop w/ detailed logging & corrected sampling ────────────────
def main():
    # Enable multithreading for PyTorch CPU kernels
    torch.set_num_threads(torch.get_num_threads())
    torch.set_num_interop_threads(torch.get_num_threads())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load demos once
    demos = load_demos()
    print(f"Loaded {len(demos)} demonstration transitions")

    # Create N_ENVS parallel envs
    venv = AsyncVectorEnv([lambda seed=i: GridEnv(seed=i) for i in range(N_ENVS)])

    # Instantiate policy & target networks
    policy_net = DuelingDQN().to(device)
    target_net = DuelingDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay    = deque(maxlen=REPLAY_SIZE)

    epsilon    = EPS_START
    step_count = 0
    rewards_hist = []

    print("🚀 Starting training…")
    for ep in trange(1, EPISODES+1, desc="Episodes"):
        obs, _      = venv.reset()           # shape (N_ENVS,3,10,10)
        ep_rewards  = np.zeros(N_ENVS)
        done_mask   = np.zeros(N_ENVS, bool)

        # Track losses and length
        total_rl_loss     = 0.0
        total_bc_loss     = 0.0
        total_margin_loss = 0.0
        num_updates       = 0
        ep_length         = 0

        for t in range(MAX_STEPS):
            ep_length = t+1

            # 1) Policy forward
            st_t = torch.from_numpy(obs).to(device)
            with torch.no_grad():
                q_vals = policy_net(st_t)    # (N_ENVS,4)

            # 2) ε-greedy action selection
            actions = []
            for i in range(N_ENVS):
                if random.random() < epsilon:
                    actions.append(random.randrange(len(ACTIONS)))
                else:
                    actions.append(torch.argmax(q_vals[i]).item())

            # 3) Step environments
            next_obs, rews, dones, truncs, _ = venv.step(actions)
            ep_rewards += rews
            done_mask  |= (dones | truncs)

            # 4) Store transitions
            for i in range(N_ENVS):
                replay.append((obs[i], actions[i], rews[i], next_obs[i], dones[i]))
            obs = next_obs

            # 5) Learning step (only if enough RL samples)
            demo_n = int(BATCH_SIZE * DEMO_RATIO)
            rl_n   = BATCH_SIZE - demo_n

            if len(replay) >= rl_n:
                # 5a) Sample RL and demo mini-batches
                rl_batch   = random.sample(replay, rl_n)
                demo_batch = random.sample(demos, demo_n)

                # 5b) Compute RL Double-DQN loss
                s_rl, a_rl, r_rl, ns_rl, d_rl = zip(*rl_batch)
                s_rl   = torch.from_numpy(np.stack(s_rl)).to(device)
                ns_rl  = torch.from_numpy(np.stack(ns_rl)).to(device)
                a_rl   = torch.tensor(a_rl, dtype=torch.long).to(device)
                r_rl   = torch.tensor(r_rl, dtype=torch.float32).to(device)
                d_rl   = torch.tensor(d_rl, dtype=torch.float32).to(device)

                q_rl    = policy_net(s_rl)
                q_next  = target_net(ns_rl).detach()
                target  = q_rl.clone()
                with torch.no_grad():
                    next_a = torch.argmax(policy_net(ns_rl), dim=1)
                for i in range(rl_n):
                    tn = q_next[i, next_a[i]].item()
                    target[i, a_rl[i]] = r_rl[i].item() + GAMMA * tn * (1.0 - d_rl[i].item())
                loss_rl = F.mse_loss(q_rl, target)

                # 5c) Behavioral cloning on demos
                s_d, a_d = zip(*demo_batch)
                s_demo = torch.from_numpy(np.stack(s_d)).to(device)
                a_demo = torch.tensor(a_d, dtype=torch.long).to(device)
                q_demo = policy_net(s_demo)
                loss_bc = F.cross_entropy(q_demo, a_demo)

                # 5d) Margin loss on demos
                q_star = q_demo.gather(1, a_demo.unsqueeze(1)).squeeze(1)
                diff   = q_demo + MARGIN - q_star.unsqueeze(1)
                mask   = torch.ones_like(diff, dtype=torch.bool)
                mask[torch.arange(demo_n), a_demo] = False
                loss_margin = F.relu(diff[mask].view(demo_n, -1)).mean()

                # 5e) Total loss & backprop
                loss = (loss_rl
                      + BC_WEIGHT   * loss_bc
                      + MARGIN_WEIGHT * loss_margin)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate for logging
                total_rl_loss     += loss_rl.item()
                total_bc_loss     += loss_bc.item()
                total_margin_loss += loss_margin.item()
                num_updates       += 1

            # 6) Sync target network
            step_count += 1
            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done_mask.all():
                break

        # ─── End‐of‐episode metrics ─────────────────────────────────────────────
        avg_reward      = ep_rewards.mean()
        success_rate    = done_mask.mean() * 100.0
        avg_rl_loss     = total_rl_loss / num_updates     if num_updates else 0.0
        avg_bc_loss     = total_bc_loss / num_updates     if num_updates else 0.0
        avg_margin_loss = total_margin_loss / num_updates if num_updates else 0.0

        rewards_hist.append(avg_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Detailed logging every 10 episodes
        if ep % 10 == 0:
            print(
                f"[Ep {ep:4d}] "
                f"R={avg_reward:6.2f} | "
                f"Succ={success_rate:5.1f}% | "
                f"Len={ep_length:3d} | "
                f"ε={epsilon:.3f} | "
                f"RL_L={avg_rl_loss:6.4f} | "
                f"BC_L={avg_bc_loss:6.4f} | "
                f"MG_L={avg_margin_loss:6.4f}"
            )

    # Save & plot
    torch.save(policy_net.state_dict(), "dqn_model.pth")
    print("✅ Training complete — model saved to dqn_model.pth")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(rewards_hist, label="Avg Reward")
    movavg = np.convolve(rewards_hist, np.ones(100)/100, mode="valid")
    plt.plot(range(100, EPISODES+1), movavg, label="100-ep MA")
    plt.xlabel("Episode"); plt.ylabel("Avg Reward")
    plt.title("Training Performance")
    plt.grid(True); plt.legend()
    plt.show()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()