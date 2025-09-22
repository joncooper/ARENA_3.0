# %%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"

import numpy as np
t.device = 'mps'
# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    # Generate some rays coming from the origin (0,0,0)
    # Return shape: num_pixels x num_points x num_dim

    # NOTE: ARENA solution uses broadcasting and slicing to add dimensions
    # rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    # t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    # rays[:, 1, 0] = 1

    origin = t.zeros(3)
    src = einops.repeat(origin, 'origin -> n origin', n=num_pixels)
    dest = t.stack([t.ones(n),
                    t.linspace(-y_limit, y_limit, num_pixels), 
                    t.zeros(num_pixels)]
                    , dim=1)
    return t.stack([src, dest], dim=1)

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# %%

def intersect_ray_1d(
    ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]
) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    # We want to test if a ray intersects a particular line segment.
    #
    # - The camera ray is defined by origin O and direction D;
    # - The object line is defined by points L_1 and L_2
    #
    # - The points on the camera ray are R(u) = O + uD for all u in [0, inf)
    # - The              object line are O(v) = L1 + v(L2 - L1) for v in [0,1]
    #
    # Setting these equal we get the solution:
    #   O + uD = L1 + v(L2 - L1)
    #   -> uD - v(L2 - L1) = L1 - O
    #   -> [Dx (L1-L2)_x, Dy (L1-L2) y].T * [u, v].T = [(L1 - O)_x, (L1 - O)_y].T
    # Once we find values of u and v that satisfy this eq, if any (the lines could be parallel) we just need to check that
    #   u >= 0 and v in [0,1]
    # .. use torch.linalg.solve and torch.stack.
    #
    #  I guess the idea is that as long as the ray and a hypothetical infinite one drawn through L1 and L2 aren't parallel (or colinear), they will eventually cross, so there has to be some solution, B, that gives us the directions along each for where they cross. But we want to make sure it happens within the segment between L1 and L2, so they both need to be not negative (in local basis terms) and not above 1 (in local basis terms

    ray = ray[:,:2]
    segment = segment[:,:2]
    L1 = segment[0]
    L2 = segment[1]
    D  = ray[1]
    A = t.stack([D, L1-L2]).T
    B = L1
    
    try:
        X = t.linalg.solve(A, B)
    except:
        return False
    
    (u, v) = X[0], X[1]
    return (u >= 0 and v >= 0 and v <= 1)

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%
ray = t.tensor([[  0.,   0.,   0.], [  1., -10.,   0.]])[:,:2]
segment = t.tensor([[  1., -12.,   0.],[  1.,  -6.,   0.]])[:,:2]


L1 = segment[0]
L2 = segment[1]
D  = ray[1]

A = t.stack([D, L1-L2]).T
B = L1

X = t.linalg.solve(A, B)
A, B, X, A @ X

# %%
fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=(0,10,1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    print(L_1, L_2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)
# %%
A = t.tensor([[6., 3.], [3., -4.]])
B = t.tensor([1., 2.])
X = t.linalg.solve(A, B)
X
# %%
A @ X
# %%
