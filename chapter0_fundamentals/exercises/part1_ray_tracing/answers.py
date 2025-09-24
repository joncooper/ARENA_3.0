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
    dest = t.stack([t.ones(num_pixels),
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
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    # rays:     shape(n, 2 = O|D,   3 = x|y|z)
    # segments: shape(n, 2 = L1|L2, 3 = x|y|z)

    O, D = rays[:,0,:-1], rays[:,1,:-1] # split origin from destination, keep only x and y (drop z)
    L1, L2 = segments[:,0,:-1], segments[:,1,:-1] # same idea
    
    assert (O.shape == D.shape) and (L1.shape == L2.shape)
    assert O.shape[-1] == 2 and L1.shape[-1] == 2

    A = D - O
    B = L1 - L2

    # A: ray_idx,     xy_dim: shape(n, 2)
    # B: segment_idx, xy_dim: shape(m, 2)
    # expand out to all pairs (A x B); both now have shape(n, m, 2) where axes are (i: row idx from A, j: row idx from B, d: coords from each row, x and y
    A_expanded = einops.repeat(A, 'ray_idx     xy_dim -> ray_idx segment_idx xy_dim', segment_idx=B.shape[0])
    B_expanded = einops.repeat(B, 'segment_idx xy_dim -> ray_idx segment_idx xy_dim', ray_idx=A.shape[0])
    assert A_expanded.shape == B_expanded.shape

    # We want o build a batch of 2x2 matrices, each is row-wise: A[i], B[j], shape overall s/b (n,m,2,2)
    # so we want (n, m, matrix row, x or y)
    matrices = t.stack([A_expanded, B_expanded], dim=-1)

    # Find cases where the ray and segment are parallel (or overlapping), and duct tape them
    #
    # NOTE: linear algebra functions in torch are batch-aware: a shape (n,m,2,2) will treat dims -2,-1 as a matrix,
    #       performing the operation and returning a tensor of shape (n,m)
    determinants = t.linalg.det(matrices)
    mask = determinants.abs() < 1e-8
    matrices[mask] = t.eye(2) # identity matrix; solver will just return the vector we're solving for

    # copy each L1, adding a dimension at the start to hold them of size = matrix's first dim
    V_exp = einops.repeat(L1, 'j d -> i j d', i=matrices.shape[0])

    x_hat = t.linalg.solve(matrices, V_exp)

    # We want to return a boolean tensor over rays, True if it intersects any segment, False else
    u, v = x_hat[..., 0], x_hat[..., 1]

    # Don't forget to eliminate any parallel cases; the solver will have worked with the
    # "duct tape to the identity matrix" trick, but we can still end up with u,v in the interval
    # if the V_exp vector itself was contained in it
    return ((u >= 0) & (v >= 0) & (v <= 1) & ~mask).any(dim=1)

tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%
def make_rays_2d(
    num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float
) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    # that is confusing.
    #
    # - there are num_pixels_y * num_pixels_z rays
    # - they all start at (0,0,0)
    # - they all end on the yz plane at (1,y,z)
    #
    # # how is that

    # # (one ray per pixel)
    # n_rays = num_pixels_y * num_pixels_z 
    # rays = t.zeros(n_rays, 2, 3)
    # # fill y on destination coordinate of all rays
    # t.linspace(-y_limit, y_limit, num_pixels_y, out=rays[:,1,1]) 
    # # fill z
    # t.linspace(-z_limit, z_limit, num_pixels_z, out=rays[:,1,2])
    # # if I've understood correctly, all x values should be set to 1;
    # # that isn't very obvious from the docstring?
    # rays[:,1,0] = 1
    # print(num_pixels_y, num_pixels_z, y_limit, z_limit, rays)
    
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)

    rays = t.zeros(n_pixels, 2, 3, dtype=t.float32)
    # all rays start from the origin and rays was zeroed so we only need to change their destinations [:][1]
    rays[:,1,0] = 1 # set all x to 1
    # rays[:,1,1] = einops.repeat(ygrid, 'y -> (y z)', z=num_pixels_z) # fill y with n_pixels_z copies of ygrid
    # rays[:,1,2] = einops.repeat(zgrid, 'z -> (y z)', y=num_pixels_y) # fill z with n_pixels_y copies of zgrid
    # ... or better -
    rays[:,1,1] = ygrid.repeat_interleave(num_pixels_z) # each element, z times
    rays[:,1,2] = zgrid.repeat(num_pixels_y)            # each array, y times

    return rays

rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
