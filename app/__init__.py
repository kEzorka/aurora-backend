import os

# Aurora's rollout allocates activations of wildly different sizes each step,
# which strands memory in the caching allocator: a 4-step run on a 32 GB V100
# died at step 4 with 9.85 GiB reserved-but-unallocated. Expandable segments
# let the allocator hand that back. Must be set before the first CUDA
# allocation, so it lives here rather than in config.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
