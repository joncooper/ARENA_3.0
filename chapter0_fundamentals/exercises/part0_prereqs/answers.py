import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"

#%%

# arr is a 4D array (the number, channels(RGB), height, width)

arr = np.load(section_dir / "numbers.npy")
print(arr[0].shape)

#%%
# 1st image in batch: (3, 150, 150); passing a 3D input plots the channels (i.e. in color)
display_array_as_img(arr[0]) 

# %%
display_array_as_img(arr[0,0]) # passing a 2D input plots in monochrome
# (in this case, the 1st channel of the 1st image in batch)

# %%
print(arr.shape)
arr_wide = einops.rearrange(arr, "b c h w -> c h (b w)")
print(arr_wide.shape)
display_array_as_img(arr_wide)

# %%
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
print(arr1.shape)
display_array_as_img(arr1)

# %%
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
display_array_as_img(arr2)

# %%
arr3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)

# %%
# Stretch vertically by factor of 2
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
display_array_as_img(arr4)

# %%
# Split channels side by side
# arr5 = einops.rearrange(arr[0], "c h w -> h (w)")
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
display_array_as_img(arr5)
# %%
# Stack 0-6 into two rows of 3
arr6 = einops.rearrange(arr, "(rows cols) c h w -> c (rows h) (cols w)", rows=2, cols=3)
display_array_as_img(arr6)

# Excellent question. That's a more advanced and concise way to achieve the same grid layout.

# Here's how einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2) works:

# "(b1 b2) c h w": This is the input pattern. It tells einops to take the first dimension of arr (the batch axis, size 6) and split it into two new axes, b1 and b2.

# b1=2: You explicitly tell einops that the length of the b1 axis should be 2. einops then infers that b2 must be 3, because the original dimension size was 6 (2 * 3 = 6). You can think of b1 as "rows" and b2 as "columns".

# "-> c (b1 h) (b2 w)": This is the output pattern.

# c: The channel axis is preserved.
# (b1 h): It creates the final image's height by stacking b1=2 images vertically. The new height is 2 * h.
# (b2 w): It creates the final image's width by laying out b2=3 images horizontally. The new width is 3 * w.
# In short, it's a compact way to say: "Take my batch of 6 images, split it into a 2-row by 3-column grid (b1=2), and then rearrange the pixels to form that grid."

# This achieves the exact same result as the previous (rows cols) example, just with different names for the temporary axes.

# %%
# Transpose x and y
arr7 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr7)
# %%
# Make the 2x3 grid as in 6, but shrink by factor of 0.5
# arr8 = einops.reduce(arr[0], 'c (h h_scale) (w w_scale) -> c h w', h_scale=2, w_scale=2, reduction='max')
arr8 = einops.reduce(arr, 
                     "(rows cols) c (h 2) (w 2) -> c (rows h) (cols w)", 
                     rows=2, cols=3, reduction='max')

display_array_as_img(arr8)

# %%
def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol=1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")

# %%
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    # "x blocks, where each is of size y"
    return einops.rearrange(t.arange(3, 9), '(rows cols) -> rows cols', cols=2)

expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)

# %%
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

        [[1, 2, 3],
     [4, 5, 6]]
    """
    return einops.rearrange(t.arange(1, 7), '(rows cols) -> rows cols', cols=3)


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))
# %%
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0

    # RECALL: "x blocks, where each is of size y"
    return einops.reduce(temps, '(n_weeks 7) -> n_weeks', reduction='mean')

temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))
# %%
# we're asking you to subtract the average temperature from each week from
# the daily temperatures. You'll have to be careful of broadcasting here,
#  since your temperatures tensor has shape (14,) while your average 
# temperature computed above has shape (2,) - these are not broadcastable.

def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    return temps - einops.repeat(temperatures_average(temps), 'weekly_avg -> (weekly_avg 7)')

expected = [-0.571, 0.429, -1.571, 3.429, -0.571, 0.429, -1.571, -4.0, 1.0, 6.0, 1.0, -1.0, -7.0, 4.0]
actual = temperatures_differences(temps)
assert_all_close(actual, t.tensor(expected))

# %%
def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    std = einops.repeat(einops.reduce(temps, '(w 7) -> w', reduction=t.std), 'w -> (w 7)')
    return temperatures_differences(temps) / std

expected = [-0.333, 0.249, -0.915, 1.995, -0.333, 0.249, -0.915, -0.894, 0.224, 1.342, 0.224, -0.224, -1.565, 0.894]
actual = temperatures_normalized(temps)
assert_all_close(actual, t.tensor(expected))

# %%
def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    # NOTE: the L2 norm of a vector is just its magnitude in Euclidean space
    return matrix / matrix.norm(dim=1, keepdim=True)

matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
assert_all_close(normalize_rows(matrix), expected)

# %%
def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.

    matrix: shape (m, n)
    """
    # The cosine similarity between two vectors is given by summing the elementwise products of the normalized vectors. 
    nr = normalize_rows(matrix)
    return nr @ nr.T    

matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[1.0, 0.975, 0.959], [0.975, 1.0, 0.998], [0.959, 0.998, 1.0]])
assert_all_close(cos_sim_matrix(matrix), expected)

# %%
def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use t.rand and t.cumsum to do this without any explicit loops.
    """
    cdf = probs.cumsum(0)
    uniform_samples = t.rand(n).unsqueeze(1) # shape: (n, 1)
    return (uniform_samples > cdf).sum(1)


n = 5_000_000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs)

#%
# Here, we're asking you to compute the accuracy of a classifier. scores is a tensor of shape (batch, n_classes) where scores[b, i] is the score the classifier gave to class i for input b, and true_classes is a tensor of shape (batch,) where true_classes[b] is the true class for input b. We want you to return the fraction of times the maximum score is equal to the true class.

# You can use the torch function t.argmax, it works as follows: tensor.argmax(dim) will return a tensor of the index containing the maximum value along the dimension dim (i.e. the shape of this output will be the same as the shape of tensor except for the dimension dim).

def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use t.argmax.
    """
    # or use .float().mean() at the end instead of the divide
    return (t.argmax(scores, dim=1) == true_classes).sum() / scores.shape[1]

scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected
print("Tests passed!")

def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    return prices[items].sum().item()

prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_indexing(prices, items) == 9.0
print("Tests passed!")

def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    """Perform a gather operation along the second dimension.

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). out[i][j] = matrix[i][indexes[i][j]]

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output. This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    """
    # YOUR CODE HERE - add assert statement(s) here for `indices` and `matrix`
    # assert(matrix.shape[0] == indexes.shape[0])
    assert(indexes.ndim == matrix.ndim)
    assert(indexes.shape[0] <= matrix.shape[0])

    out = matrix.gather(1, indexes)
    assert(out.shape == indexes.shape)

    return out

matrix = t.arange(15).view(3, 5)
indexes = t.tensor([[4], [3], [2]])
expected = t.tensor([[4], [8], [12]])
assert_all_equal(gather_2d(matrix, indexes), expected)

indexes2 = t.tensor([[2, 4], [1, 3], [0, 2]])
expected2 = t.tensor([[2, 4], [6, 8], [10, 12]])
assert_all_equal(gather_2d(matrix, indexes2), expected2)

def total_price_gather(prices: Tensor, items: Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather."""
    assert items.max() < prices.shape[0]
    return t.gather(prices, 0, items).sum()



prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_gather(prices, items) == 9.0
print("Tests passed!")

def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )
    """
    # coords is 1 row per lookup w/ 2 indices in that row
    # matrix is a 2d of values, you need to look up w/ those
    # to do this indexing we need to pass indices per-dim
    # note that d_n need not == n to work; n <= d_n suffices

    return matrix[*coords.T]

mat_2d = t.arange(15).view(3, 5)
coords_2d = t.tensor([[0, 1], [0, 4], [1, 4]])
actual = integer_array_indexing(mat_2d, coords_2d)
assert_all_equal(actual, t.tensor([1, 4, 9]))

mat_3d = t.arange(2 * 3 * 4).view((2, 3, 4))
coords_3d = t.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [1, 2, 0]])
actual = integer_array_indexing(mat_3d, coords_3d)
assert_all_equal(actual, t.tensor([0, 5, 10, 15, 20]))

# %%
def batched_logsumexp(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    # TODO: work through the blog posts; for now, follow NumPy version from 2nd one
    # def logsumexp(x):
    #   c = x.max()
    #   return c + np.log(np.sum(np.exp(x - c)))

    a = matrix.max(dim=1, keepdim=True).values
    centered = matrix - a
    # The squeeze here is duct-tape - every couple years I get to re-learn
    # NumPy and go through figuring out the dimensional idioms/behavior
    # again. This is the 2025 iteration I guess. Woo!

    # (for comparison - ARENA's implementation):
    # C = matrix.max(dim=-1).values
    # exps = t.exp(matrix - einops.rearrange(C, "n -> n 1"))
    # return C + t.log(t.sum(exps, dim=-1))

    return a.squeeze() + t.exp(centered).sum(dim=1).log()

matrix = t.tensor([[-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]])
expected = t.tensor([-1000 + math.log(4), 1000 + math.log(4)])
actual = batched_logsumexp(matrix)
assert_all_close(actual, expected)

matrix2 = t.randn((10, 20))
expected2 = t.logsumexp(matrix2, dim=-1)
actual2 = batched_logsumexp(matrix2)
assert_all_close(actual2, expected2)

# %%

def batched_softmax(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute softmax(row).

    Do this without using PyTorch's softmax function.
    Instead, use the definition of softmax: https://en.wikipedia.org/wiki/Softmax_function

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.
    """
    # def: softmax takes any list of real numbers and turns it into a probability distribution
    # $ p_i = {e^x_i}/{sum{j}{e^x_j}} $
    
    numerator = matrix.exp()
    denominator = t.sum(numerator, dim=-1, keepdim=True)
    return numerator / denominator 

# %%
matrix = t.arange(1, 6).view((1, 5)).float().log()
expected = t.arange(1, 6).view((1, 5)) / 15.0
actual = batched_softmax(matrix)
assert_all_close(actual, expected)
for i in [0.12, 3.4, -5, 6.7]:
    assert_all_close(actual, batched_softmax(matrix + i))  # check it's translation-invariant

# %%
matrix2 = t.rand((10, 20))
actual2 = batched_softmax(matrix2)
assert actual2.min() >= 0.0
assert actual2.max() <= 1.0
assert_all_equal(actual2.argsort(), matrix2.argsort())
assert_all_close(actual2.sum(dim=-1), t.ones(matrix2.shape[:-1]))

def batched_logsoftmax(matrix: Tensor) -> Tensor:
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n).

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    """
    # Compare w/ ARENA's implementation:
    #  C = matrix.max(dim=1, keepdim=True).values
    # return matrix - C - (matrix - C).exp().sum(dim=1, keepdim=True).log()

    # Apparently log(softmax(row)) == x - logsumexp(x), which, ok? TODO: verify
    return matrix - batched_logsumexp(matrix).unsqueeze(1)


# %%

matrix = t.arange(1, 7).view((2, 3)).float()
start = 1000
matrix2 = t.arange(start + 1, start + 7).view((2, 3)).float()
actual = batched_logsoftmax(matrix2)
expected = t.tensor([[-2.4076, -1.4076, -0.4076],
                     [-2.4076, -1.4076, -0.4076]])
assert_all_close(actual, expected)
# %%

def batched_cross_entropy_loss(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Compute the cross entropy loss for each example in the batch.

    logits: shape (batch, classes). logits[i][j] is the unnormalized prediction for example i and class j.
    true_labels: shape (batch, ). true_labels[i] is an integer index representing the true class for example i.

    Return: shape (batch, ). out[i] is the loss for example i.

    Hint: convert the logits to log-probabilities using your batched_logsoftmax from above.
    Then the loss for an example is just the negative of the log-probability that the model assigned to the true class. Use torch.gather to perform the indexing.
    """
    assert logits.shape[0] == true_labels.shape[0]
    assert true_labels.max() < logits.shape[1]

    # Compare ARENA version:
    #     logprobs = batched_logsoftmax(logits)
    #     indices = einops.rearrange(true_labels, "n -> n 1")
    #     pred_at_index = logprobs.gather(1, indices)
    #     return -einops.rearrange(pred_at_index, "n 1 -> n")

    neg_log_probs = batched_logsoftmax(logits) * -1.0
    return t.gather(neg_log_probs, 1, true_labels.unsqueeze(1)).squeeze()

# %%

logits = t.tensor([[float("-inf"), float("-inf"), 0], [1 / 3, 1 / 3, 1 / 3], [float("-inf"), 0, 0]])
true_labels = t.tensor([2, 0, 0])
expected = t.tensor([0.0, math.log(3), float("inf")])
actual = batched_cross_entropy_loss(logits, true_labels)
assert_all_close(actual, expected)

# %%
def collect_rows(matrix: Tensor, row_indexes: Tensor) -> Tensor:
    """Return a 2D matrix whose rows are taken from the input matrix in order according to row_indexes.

    matrix: shape (m, n)
    row_indexes: shape (k,). Each value is an integer in [0..m).

    Return: shape (k, n). out[i] is matrix[row_indexes[i]].
    """
    return matrix[row_indexes]

# %%
matrix = t.arange(15).view((5, 3))
row_indexes = t.tensor([0, 2, 1, 0])
actual = collect_rows(matrix, row_indexes)
expected = t.tensor([[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 1, 2]])
assert_all_equal(actual, expected)
# %%
def collect_columns(matrix: Tensor, column_indexes: Tensor) -> Tensor:
    """Return a 2D matrix whose columns are taken from the input matrix in order according to column_indexes.

    matrix: shape (m, n)
    column_indexes: shape (k,). Each value is an integer in [0..n).

    Return: shape (m, k). out[:, i] is matrix[:, column_indexes[i]].
    """
    assert column_indexes.max() < matrix.shape[1]
    return matrix[:,column_indexes]

# %%

matrix = t.arange(15).view((5, 3))
column_indexes = t.tensor([0, 2, 1, 0])
actual = collect_columns(matrix, column_indexes)
expected = t.tensor([[0, 2, 1, 0], [3, 5, 4, 3], [6, 8, 7, 6], [9, 11, 10, 9], [12, 14, 13, 12]])
assert_all_equal(actual, expected)

# %%

def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    assert(mat.shape[0] == mat.shape[1])
    # trace is the sum of the diagonal elements in a matrix
    return einops.einsum(mat * np.eye(3), 'rows cols -> ')

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    # Straightforward enough but still feels way too magical
    return einops.einsum(mat, vec, 'i j, j -> i')

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    # mat1: p x q
    # mat2: q x r
    return einops.einsum(mat1, mat2, 'p q, q r -> p r')

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.inner`.
    """
    # u = [a_1, a_2, .., a_N]; v = [b_1, b_2, .., a_N]
    # u dot v = a_1 * b_1 + a_2 * b_2 + ... + a_N * b_N
    # u: m x 1
    # v: 1 x m
    return einops.einsum(vec1, vec2, 'm, m ->')

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.outer`.
    """
    # u: m x 1
    # v: n x 1
    # u <outer> v = u1v1 + u1v2 + ... + u1v[N-1]
    #               u2v1 + u2v2 + ... + u2v[N-1]
    #               ................. + u[M-1][N-1]
    # u: m x n ?
    return einops.einsum(vec1, vec2, 'm, n -> m n')


tests.test_einsum_trace(einsum_trace)
tests.test_einsum_mv(einsum_mv)
tests.test_einsum_mm(einsum_mm)
tests.test_einsum_inner(einsum_inner)
tests.test_einsum_outer(einsum_outer)


# %%
