# path_robot.py
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

# ─── RL AGENT STUB ─────────────────────────────────────────────────────────────
def find_path(obstacles: set[tuple[int,int]], start: tuple[int,int], goal: tuple[int,int]):
    """
    Dummy path-finder: straight line until blocked by an obstacle.
    Replace with your trained RL agent’s inference.
    - obstacles: set of (y,x) tuples
    - start, goal: (y,x) tuples
    Returns: list of (y,x) waypoints
    """
    path = []
    y0, x0 = start
    y1, x1 = goal
    steps = max(abs(y1 - y0), abs(x1 - x0))
    for i in range(steps + 1):
        y = int(y0 + (y1 - y0) * i / steps)
        x = int(x0 + (x1 - x0) * i / steps)
        if (y, x) in obstacles:
            break
        path.append((y, x))
    return path

# ─── STREAMLIT UI ──────────────────────────────────────────────────────────────
st.title("RL Path Planner (Cell Selector)")

# Sidebar controls
GRID     = st.sidebar.slider("Grid size", 5, 30, 10)
CELL_PX  = st.sidebar.slider("Cell size (px)", 20, 40, 30)
START    = tuple(st.sidebar.slider("Start (y, x)", 0, GRID-1, (0, 0)))
GOAL     = tuple(st.sidebar.slider("Goal  (y, x)", 0, GRID-1, (GRID-1, GRID-1)))
compute  = st.sidebar.button("Compute Path")

# Initialize or reset the mask in session state
if "mask" not in st.session_state or st.session_state.mask.shape != (GRID, GRID):
    st.session_state.mask = np.zeros((GRID, GRID), dtype=bool)

# Render a GRID×GRID of checkboxes for obstacle selection
st.markdown("#### Draw Obstacles by Ticking Cells")
for y in range(GRID):
    cols = st.columns(GRID)
    for x, col in enumerate(cols):
        with col:
            checked = st.checkbox(
                label="",
                key=f"cell_{y}_{x}",
                value=st.session_state.mask[y, x]
            )
            st.session_state.mask[y, x] = checked

# Convert mask to 0/1 array for display
mask = st.session_state.mask.astype(int)
st.markdown("**Current Obstacle Mask**")
st.write(mask)

# Build the set of obstacle coordinates
OBSTACLES = {
    (y, x)
    for y in range(GRID)
    for x in range(GRID)
    if mask[y, x] == 1
}

# Compute and draw once user clicks
if compute:
    path = find_path(OBSTACLES, START, GOAL)

    # Create canvas image
    img = Image.new("RGB", (GRID * CELL_PX, GRID * CELL_PX), "white")
    draw = ImageDraw.Draw(img)

    # Draw grid lines
    for i in range(GRID + 1):
        # vertical
        draw.line([(i * CELL_PX, 0), (i * CELL_PX, GRID * CELL_PX)], fill="gray")
        # horizontal
        draw.line([(0, i * CELL_PX), (GRID * CELL_PX, i * CELL_PX)], fill="gray")

    # Fill obstacles
    for (y, x) in OBSTACLES:
        xy = [
            x * CELL_PX,
            y * CELL_PX,
            (x + 1) * CELL_PX,
            (y + 1) * CELL_PX,
        ]
        draw.rectangle(xy, fill="black")

    # Draw planned path
    if path:
        pts = [
            (x * CELL_PX + CELL_PX // 2, y * CELL_PX + CELL_PX // 2)
            for (y, x) in path
        ]
        draw.line(pts, fill="red", width=3)

    st.image(img, caption="Planned Path", use_column_width=True)
