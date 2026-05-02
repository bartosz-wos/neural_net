# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Queue (bottom = next to pop)

- **<empty — no pending items>**

## Done

- **DataLoader / Dataset infrastructure**: Implement MNIST-style mini-batch loader with shuffle, batch iteration, TensorDataset wrapper, and integrate into Model::train() method. Add DataLoader class that wraps a Dataset and yields mini-batches. Implement a Trainer class that handles epoch-level training loops with DataLoader support.

## Done

- **FlashAttention-style memory-efficient attention**: Implement memory-efficient multi-head attention with tiled computation, online softmax normalization, and exact gradient computation. Include FlashAttentionLayer class with forward/backward passes that compute attention in O(sequence_length) memory instead of O(sequence_length^2). Add CPU tiling fallback for non-AVX systems. Benchmark against standard attention for correctness and memory savings.
