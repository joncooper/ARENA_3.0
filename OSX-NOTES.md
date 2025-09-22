We can't promise everything will run locally on MacOS, and recommend you try one of the other methods mentioned in this section. If you do try using MacOS:

- Be sure to set device='mps'.
- Certain operations like torch.linalg.det and torch.linalg.solve aren't available in PyTorch's MPS backend yet. You can enable CPU fallback for these operations by setting the environmental variable PYTORCH_ENABLE_MPS_FALLBACK=1.
- PyTorch DataLoaders with num_workers > 1 can cause Python to run out of memory and crash on MacOS! To fix this, add the argument multiprocessing_context="fork" to any DataLoaders before running the cells containing them.
