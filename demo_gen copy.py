# generate_demos.py
# Build demonstration transitions on corner-case obstacle maps,
# visualize each map with obstacles in red, BFS path in thicker green,
# grid lines visible, and start/end labeled.

import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# ─── Constants ────────────────────────────────────────────────────────────────
GRID_SIZE  = 10
START      = (0, 0)
GOAL       = (9, 9)
ACTIONS    = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
DEMO_PATH  = "demos.pt"

# ─── BFS Utility ──────────────────────────────────────────────────────────────
def bfs_path(obstacles, start, goal):
    """Return shortest path list from start to goal avoiding obstacles."""
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        if path[-1] == goal:
            return path
        r, c = path[-1]
        for dy, dx in ACTIONS:
            nxt = (r + dy, c + dx)
            if (0 <= nxt[0] < GRID_SIZE and
                0 <= nxt[1] < GRID_SIZE and
                nxt not in obstacles and
                nxt not in visited):
                visited.add(nxt)
                queue.append(path + [nxt])
    return []

# ─── Define 10 obstacle maps ───────────────────────────────────────────────────
demo_obstacle_sets = []
# 1) Horizontal wall at row 5, gap at col 0
demo_obstacle_sets.append({(5, x) for x in range(GRID_SIZE) if x != 0})
# 2) Vertical wall at col 5, gap at row 8
demo_obstacle_sets.append({(y, 5) for y in range(GRID_SIZE) if y != 8})
# 3) Snake barrier: zig-zag across rows requiring winding path
demo_obstacle_sets.append(
    {(1, x) for x in range(GRID_SIZE) if x != GRID_SIZE-1} |
    {(3, x) for x in range(GRID_SIZE) if x != 0} |
    {(5, x) for x in range(GRID_SIZE) if x != GRID_SIZE-1} |
    {(7, x) for x in range(GRID_SIZE) if x != 0}
)
# 4) Two vertical barriers: col 3 blocked except bottom, col 6 except top
demo_obstacle_sets.append(
    {(r, 3) for r in range(GRID_SIZE) if r != GRID_SIZE - 1} |
    {(r, 6) for r in range(GRID_SIZE) if r != 0}
)
# 5) Both holes at bottom in col 3 and col 6
demo_obstacle_sets.append(
    {(r, 3) for r in range(GRID_SIZE) if r != GRID_SIZE - 1} |
    {(r, 5) for r in range(3,10)} |
    {(r, 7) for r in range(1,10)}
)
# 6) Both holes at top in col 3 and col 6
demo_obstacle_sets.append(
    {(r, 3) for r in range(GRID_SIZE) if r != 0} |
    {(r, 6) for r in range(GRID_SIZE) if r != 0} 
)
# 7) Spiral ring barrier around center with two openings
tmp_ring = (
    {(3, c) for c in range(GRID_SIZE)} |
    {(6, c) for c in range(GRID_SIZE)} |
    {(r, 2) for r in range(GRID_SIZE)} |
    {(r, 7) for r in range(GRID_SIZE)}
)
# remove openings at (4,2) and (7,5)
tmp_ring -= {(4, 2), (4, 7), (3, 0), (6, 9) }
demo_obstacle_sets.append(tmp_ring)
# 8) Zigzag wall: alternating segments in rows 2-7 creating winding path
demo_obstacle_sets.append(
    {(2, c) for c in range(1, 9)} |
    {(3, 1)} |
    {(4, c) for c in range(1, 9)} |
    {(5, 8)} |
    {(6, c) for c in range(1, 9)} |
    {(7, 1)}
)
# 9) Dual 3×3 blocks near start and goal forcing detours
block1 = {(r, c) for r in range(1, 4) for c in range(0, 4)}
block2 = {(r, c) for r in range(6, 10) for c in range(6, 9)}
demo_obstacle_sets.append(block1 | block2)
# 10) Spiral ring barrier around center with two openings
tmp_ring = (
    {(3, c) for c in range(GRID_SIZE)} |
    {(6, c) for c in range(GRID_SIZE)} |
    {(r, 2) for r in range(GRID_SIZE)} |
    {(r, 7) for r in range(GRID_SIZE)}
)
# remove openings at (4,2) and (7,5)
tmp_ring -= {(5, 2), (3, 4), (3, 0), (3, 9) ,(1, 7), (6, 9) }
demo_obstacle_sets.append(tmp_ring)
# ─── Build demos and collect for visualization ────────────────────────────────
demos = []
valid_maps = []
valid_paths = []
for i, obstacles in enumerate(demo_obstacle_sets, start=1):
    path = bfs_path(obstacles, START, GOAL)
    if not path:
        print(f"Skipping map #{i}: no valid path.")
        continue
    print(f"Map #{i}: path length {len(path)}")
    valid_maps.append(obstacles)
    valid_paths.append(path)
    # convert to demo transitions
    for s0, s1 in zip(path, path[1:]):
        agent  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        goal_m = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        obs_m  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        agent[s0]    = 1.0
        goal_m[GOAL] = 1.0
        for cell in obstacles:
            obs_m[cell] = 1.0
        obs_np = np.stack([agent, goal_m, obs_m], axis=0)
        move = (s1[0] - s0[0], s1[1] - s0[1])
        demos.append((obs_np, ACTION_IDX[move]))

print(f"Saving {len(demos)} demo transitions to {DEMO_PATH}")
torch.save(demos, DEMO_PATH)

# ─── Visualization ───────────────────────────────────────────────────────────
rows = int(np.ceil(len(valid_maps) / 3))
cols = min(3, len(valid_maps))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = axes.flatten() if len(valid_maps) > 1 else [axes]

for idx, (obstacles, path) in enumerate(zip(valid_maps, valid_paths)):
    ax = axes[idx]
    # white background
    ax.imshow(np.ones((GRID_SIZE, GRID_SIZE, 3)), origin='upper')

    # explicit grid lines
    for x in range(GRID_SIZE + 1):
        ax.axvline(x, color='gray', linewidth=1, zorder=1)
    for y in range(GRID_SIZE + 1):
        ax.axhline(y, color='gray', linewidth=1, zorder=1)

    # plot obstacles in red
    for (r, c) in obstacles:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color='red', zorder=2))

    # plot BFS path in thicker green line
    xs = [c + 0.5 for (r, c) in path]
    ys = [r + 0.5 for (r, c) in path]
    ax.plot(xs, ys, color='green', linewidth=3, zorder=3)

    # label start/end
    ax.text(START[1] + 0.5, START[0] + 0.5, 'S', ha='center', va='center',
            color='black', fontsize=14, weight='bold', zorder=4)
    ax.text(GOAL[1] + 0.5, GOAL[0] + 0.5, 'G', ha='center', va='center',
            color='black', fontsize=14, weight='bold', zorder=4)

    ax.set_title(f"Demo Map #{idx+1}")
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, 0)
    ax.axis('off')

# hide unused axes
for j in range(len(valid_maps), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()