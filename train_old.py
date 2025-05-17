# train.py
# Dueling Double‑DQN training with reward shaping, slower ε‑decay,
# GPU acceleration, multithreading, and Gym’s vectorized environments.
# Safe for Windows multiprocessing with spawn.

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

import gym
from gym import spaces
from gym.vector import AsyncVectorEnv


# ─── 1) ENVIRONMENT DEFINITION ────────────────────────────────────────────────
GRID_SIZE = 10
START     = (0, 0)
GOAL      = (GRID_SIZE - 1, GRID_SIZE - 1)
ACTIONS   = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NUM_WALLS = 15  # obstacles per episode

class GridEnv(gym.Env):
    """Custom grid with random obstacles and reward shaping."""
    metadata = {'render.modes': []}

    def __init__(self, seed=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, GRID_SIZE, GRID_SIZE),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        if seed is not None:
            random.seed(seed)

    def reset(self):
        # sample obstacles, reset state
        cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        cells.remove(START); cells.remove(GOAL)
        self.obstacles = set(random.sample(cells, NUM_WALLS))
        self.state     = START
        obs = self._get_obs()
        return obs, {}  # must return (obs, info)

    def step(self, action):
        dy, dx = ACTIONS[action]
        y, x   = self.state
        nxt    = (y + dy, x + dx)

        # reward shaping: Manhattan distance improvement
        dist_old = abs(y - GOAL[0]) + abs(x - GOAL[1])
        dist_new = abs(nxt[0] - GOAL[0]) + abs(nxt[1] - GOAL[1])

        if not (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE) or nxt in self.obstacles:
            reward = -10
            nxt = self.state
        elif nxt == GOAL:
            reward = 20
        else:
            reward = -1 + 0.1 * (dist_old - dist_new)

        done = (nxt == GOAL)
        self.state = nxt
        obs = self._get_obs()
        # new API: return (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, {}

    def _get_obs(self):
        agent = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        goal  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        obs   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        agent[self.state] = 1.0
        goal[GOAL]        = 1.0
        for (r, c) in self.obstacles:
            obs[r, c]   = 1.0
        return np.stack([agent, goal, obs], axis=0)


# ─── 2) DUELING DOUBLE‑DQN NETWORK ─────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
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
        return val + adv - adv.mean(dim=1, keepdim=True)


# ─── 3) MAIN TRAINING FUNCTION ────────────────────────────────────────────────
def main():
    # multithread BLAS/OpenMP
    torch.set_num_threads(torch.get_num_threads())
    torch.set_num_interop_threads(torch.get_num_threads())

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # vectorized envs
    N_ENVS = 4
    def make_env(seed):
        return lambda: GridEnv(seed)
    venv = AsyncVectorEnv([make_env(i) for i in range(N_ENVS)])

    # networks & optimizer
    policy_net = DuelingDQN().to(device)
    target_net = DuelingDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()
    replay    = deque(maxlen=5000)

    epsilon    = 1.0
    rewards_hist = []
    step_count = 0

    # training loop
    for ep in trange(1, 5001, desc="Episodes"):
        # reset returns obs, info
        obs, _ = venv.reset()  # obs shape: (N_ENVS,3,10,10)
        ep_rewards = np.zeros(N_ENVS)
        done_all   = np.zeros(N_ENVS, bool)

        for _ in range(100):
            st_t = torch.from_numpy(obs).to(device)
            with torch.no_grad():
                q_vals = policy_net(st_t)  # (N_ENVS,4)

            # ε-greedy
            actions = []
            for i in range(N_ENVS):
                if random.random() < epsilon:
                    actions.append(random.randrange(len(ACTIONS)))
                else:
                    actions.append(torch.argmax(q_vals[i]).item())

            next_obs, rews, dones, truncs, _ = venv.step(actions)
            ep_rewards += rews
            done_all |= np.array(dones) | np.array(truncs)

            # store transitions
            for i in range(N_ENVS):
                replay.append((obs[i], actions[i], rews[i], next_obs[i], dones[i]))

            obs = next_obs

            # learning
            if len(replay) >= 64:
                batch = random.sample(replay, 64)
                s_b, a_b, r_b, ns_b, d_b = zip(*batch)
                s_b  = torch.from_numpy(np.stack(s_b)).to(device)
                ns_b = torch.from_numpy(np.stack(ns_b)).to(device)
                a_b  = torch.tensor(a_b, dtype=torch.long).to(device)
                r_b  = torch.tensor(r_b, dtype=torch.float32).to(device)
                d_b  = torch.tensor(d_b, dtype=torch.float32).to(device)

                q      = policy_net(s_b)
                q_next = target_net(ns_b).detach()
                target = q.clone()

                # Double-DQN update
                with torch.no_grad():
                    next_a = torch.argmax(policy_net(ns_b), dim=1)
                for i in range(64):
                    tn = q_next[i, next_a[i]].item()
                    target[i, a_b[i]] = r_b[i].item() + 0.99 * tn * (1.0 - d_b[i].item())

                loss = loss_fn(q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # sync target
            step_count += 1
            if step_count % 200 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done_all.all():
                break

        rewards_hist.append(ep_rewards.mean())
        epsilon = max(0.05, epsilon * 0.9995)
        if ep % 100 == 0:
            print(f"[Episode {ep}] AvgReward={rewards_hist[-1]:.2f}, ε={epsilon:.3f}")

    # save & plot
    torch.save(policy_net.state_dict(), "dqn_model.pth")
    print("Training complete — model saved to dqn_model.pth")

    plt.figure(figsize=(8,5))
    plt.plot(rewards_hist, label="Avg Reward")
    ma = np.convolve(rewards_hist, np.ones(100)/100, mode="valid")
    plt.plot(range(100,5001), ma, label="100-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.grid(True)
    plt.show()


# ─── 4) ENTRY POINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
