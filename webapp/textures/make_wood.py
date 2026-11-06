"""Procedural wooden plank floor texture, seamlessly tileable."""
import sys
import numpy as np
from PIL import Image, ImageFilter
rng = np.random.default_rng(7)
S = 1024                 # one texture tile
PLANKS = 5               # plank rows per tile
edges = np.linspace(0, S, PLANKS + 1).astype(int)
out = np.zeros((S, S, 3), float)
X = np.arange(S, dtype=float)
base_colors = np.array([[181, 131, 84], [166, 116, 72], [192, 142, 93], [155, 107, 66], [174, 124, 77]], float)
fine = rng.standard_normal((S, S))
for r in range(PLANKS):
    y0, y1 = edges[r], edges[r + 1]
    yy = np.arange(y1 - y0, dtype=float)[:, None]
    offset = int(rng.integers(0, S))
    for k in range(1):                       # one plank per row per tile
        start = offset
        cols = np.ones(S, bool)
        color = base_colors[rng.integers(len(base_colors))] * rng.uniform(0.95, 1.05)
        # grain lines run along x; gently bent by a few periodic waves
        bend = sum(rng.uniform(2, 6) / (i + 1) * np.sin(2 * np.pi * rng.integers(1, 4) * X[cols] / S + rng.uniform(0, 6.3))
                   for i in range(3))
        density = rng.uniform(0.25, 0.4)
        g = np.sin(density * (yy + bend[None, :]) + 0.8 * np.sin(0.05 * yy))
        g = (0.5 + 0.5 * g) ** 4             # thin dark lines
        shade = 1 - 0.13 * g + 0.025 * fine[y0:y1][:, cols]
        out[y0:y1, cols] = color * shade[..., None]
        out[y0:y1, start:start + 2] *= 0.6   # end joint
    out[y0:y0 + 2, :] *= 0.6                 # gap between rows
img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
img.save(sys.argv[1])
