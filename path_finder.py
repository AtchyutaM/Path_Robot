# app.py
# Streamlit UI for RL Path Planner, importing the find_path function from a separate module

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import math

# Import the RL sub-agent
from findpath import find_path

# ---- Streamlit App ----
st.title("RL Path Planner Demo")
st.markdown("Select grid cells to place obstacles, then click 'Find Path'.")

# Sidebar controls
GRID     = st.sidebar.slider("Grid size", 5, 30, 10)
CELL_PX  = st.sidebar.slider("Cell size (px)", 20, 40, 30)
START    = tuple(st.sidebar.slider("Start (y, x)", 0, GRID-1, (0, 0)))
GOAL     = tuple(st.sidebar.slider("Goal  (y, x)", 0, GRID-1, (GRID-1, GRID-1)))
FIND_BTN = st.sidebar.button("Find Path")

# Initialize or reset mask
if "mask" not in st.session_state or st.session_state.mask.shape != (GRID, GRID):
    st.session_state.mask = np.zeros((GRID, GRID), dtype=bool)

# Render grid of checkboxes for obstacles
st.markdown("### Obstacle Grid")
for y in range(GRID):
    cols = st.columns(GRID)
    for x, col in enumerate(cols):
        with col:
            checked = st.checkbox("", key=f"cell_{y}_{x}", value=st.session_state.mask[y, x])
            st.session_state.mask[y, x] = checked

# Display raw mask
mask = st.session_state.mask.astype(int)
st.markdown("**Current Obstacle Mask**")
st.write(mask)

# When user clicks Find Path
if FIND_BTN:
    # Build obstacles set of (y,x)
    obstacles = {(y, x) for y in range(GRID) for x in range(GRID) if mask[y, x] == 1}
    # Call RL sub-agent
    path = find_path(obstacles, START, GOAL)

    # Create drawing canvas
    img = Image.new("RGB", (GRID * CELL_PX, GRID * CELL_PX), "white")
    draw = ImageDraw.Draw(img)

    # Draw grid lines
    for i in range(GRID + 1):
        draw.line([(i*CELL_PX, 0), (i*CELL_PX, GRID*CELL_PX)], fill="gray")
        draw.line([(0, i*CELL_PX), (GRID*CELL_PX, i*CELL_PX)], fill="gray")

    # Draw obstacles in red
    for (y, x) in obstacles:
        rect = [x*CELL_PX, y*CELL_PX, (x+1)*CELL_PX, (y+1)*CELL_PX]
        draw.rectangle(rect, fill="red")

    # Draw path in blue with arrows
    if path and len(path) > 1:
        arrow_size = CELL_PX * 0.3
        for (y0, x0), (y1, x1) in zip(path, path[1:]):
            x0p, y0p = x0*CELL_PX + CELL_PX/2, y0*CELL_PX + CELL_PX/2
            x1p, y1p = x1*CELL_PX + CELL_PX/2, y1*CELL_PX + CELL_PX/2
            # line segment
            draw.line([(x0p, y0p), (x1p, y1p)], fill="blue", width=3)
            # arrowhead
            angle = math.atan2(y1p - y0p, x1p - x0p)
            left = (x1p - arrow_size*math.cos(angle - math.pi/6),
                    y1p - arrow_size*math.sin(angle - math.pi/6))
            right = (x1p - arrow_size*math.cos(angle + math.pi/6),
                     y1p - arrow_size*math.sin(angle + math.pi/6))
            draw.polygon([(x1p, y1p), left, right], fill="blue")
    elif path:
        # single-cell path: mark with blue dot
        y, x = path[0]
        cx, cy = x*CELL_PX + CELL_PX/2, y*CELL_PX + CELL_PX/2
        r = CELL_PX * 0.2
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill="blue")

    # Display result
    st.image(img, caption="Obstacles (red) and Path (blue)", use_column_width=True)
