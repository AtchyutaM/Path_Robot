# generate_demos.py
import random
import pickle
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

# ─── Constants ────────────────────────────────────────────────────────────────
GRID_SIZE   = 10
START       = (0, 0)
GOAL        = (9, 9)
ACTIONS     = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_IDX  = {a: i for i, a in enumerate(ACTIONS)}
DEMO_PATH   = "demos.pt"

# ─── BFS Utility ──────────────────────────────────────────────────────────────
def bfs_path(obstacles, start, goal):
    queue, seen = deque([[start]]), {start}
    while queue:
        path = queue.popleft()
        if path[-1] == goal:
            return path
        r, c = path[-1]
        for dy, dx in ACTIONS:
            nxt = (r + dy, c + dx)
            if (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE
                    and nxt not in obstacles and nxt not in seen):
                seen.add(nxt)
                queue.append(path + [nxt])
    return []

# ─── 10 Base Obstacle Sets ────────────────────────────────────────────────────
base_sets = [
    {(5, x) for x in range(GRID_SIZE) if x != 0},
    {(y, 5) for y in range(GRID_SIZE) if y != 8},
    {(1, x) for x in range(GRID_SIZE) if x != GRID_SIZE-1} |
    {(3, x) for x in range(GRID_SIZE) if x != 0} |
    {(5, x) for x in range(GRID_SIZE) if x != GRID_SIZE-1} |
    {(7, x) for x in range(GRID_SIZE) if x != 0},
    {(r, 3) for r in range(GRID_SIZE) if r != GRID_SIZE-1} |
    {(r, 6) for r in range(GRID_SIZE) if r != 0},
    {(r, 3) for r in range(GRID_SIZE) if r != GRID_SIZE-1} |
    {(r, 5) for r in range(3,10)} |
    {(r, 7) for r in range(1,10)},
    {(r, 3) for r in range(GRID_SIZE) if r != 0} |
    {(r, 6) for r in range(GRID_SIZE) if r != 0},
    ({(3, c) for c in range(2, 8)} |
     {(6, c) for c in range(2, 8)} |
     {(r, 2) for r in range(2, 8)} |
     {(r, 7) for r in range(2, 8)}) - {(4,2),(7,5)},
    {(2, c) for c in range(1, 9)} | {(3,1)} | {(4, c) for c in range(1, 9)} |
    {(5,8)} | {(6, c) for c in range(1, 9)} | {(7,1)},
    {(r, c) for r in range(1, 4) for c in range(0, 4)} |
    {(r, c) for r in range(6,10) for c in range(6, 9)},
    ({(3, c) for c in range(GRID_SIZE)} |
     {(6, c) for c in range(GRID_SIZE)} |
     {(r, 2) for r in range(GRID_SIZE)} |
     {(r, 7) for r in range(GRID_SIZE)}) -
    {(5,2),(3,4),(3,0),(3,9),(1,7),(6,9)}
]

# ─── Generate 10 Variations per Base ──────────────────────────────────────────
all_variations = []
cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
         if (r, c) not in (START, GOAL)]
for base in base_sets:
    variants = []
    while len(variants) < 10:
        new_obs = set(base)
        # perform 3 random swaps
        for _ in range(3):
            if not new_obs: break
            rem = random.choice(tuple(new_obs))
            new_obs.remove(rem)
            add = random.choice([cell for cell in cells if cell not in new_obs])
            new_obs.add(add)
        # accept if solvable and unique
        if bfs_path(new_obs, START, GOAL) and all(new_obs != v for v in variants):
            variants.append(new_obs)
    all_variations.append(variants)

# ─── Build & Save Demo Transitions ───────────────────────────────────────────
demos = []
for variants in all_variations:
    for obstacles in variants:
        path = bfs_path(obstacles, START, GOAL)
        for s0, s1 in zip(path, path[1:]):
            agent = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
            goal_m = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
            obs_m = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
            agent[s0] = 1.0
            goal_m[GOAL] = 1.0
            for cell in obstacles:
                obs_m[cell] = 1.0
            obs_np = np.stack([agent, goal_m, obs_m], axis=0)
            action = (s1[0]-s0[0], s1[1]-s0[1])
            demos.append((obs_np, ACTION_IDX[action]))

print(f"Generated {len(demos)} transitions total")

torch.save(demos, DEMO_PATH)

# ─── Visualization: 10×10 Grid of Variations ─────────────────────────────────
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i, variants in enumerate(all_variations):
    for j, obstacles in enumerate(variants):
        ax = axes[i][j]
        ax.imshow(np.ones((GRID_SIZE, GRID_SIZE, 3)), origin='upper')
        # grid lines
        for x in range(GRID_SIZE+1):
            ax.axvline(x, color='gray', linewidth=0.5)
            ax.axhline(x, color='gray', linewidth=0.5)
        # obstacles
        for (r, c) in obstacles:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color='red'))
        # path
        path = bfs_path(obstacles, START, GOAL)
        xs = [c + 0.5 for _, c in path]
        ys = [r + 0.5 for r, _ in path]
        ax.plot(xs, ys, color='green', linewidth=2)
        # labels
        ax.text(START[1]+0.5, START[0]+0.5, 'S',
                ha='center', va='center', color='black')
        ax.text(GOAL[1]+0.5, GOAL[0]+0.5, 'G',
                ha='center', va='center', color='black')
        ax.axis('off')

plt.tight_layout()
plt.show()
