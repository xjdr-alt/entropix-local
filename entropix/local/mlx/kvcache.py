import mlx.core as mx
import mlx.nn as nn

class KVCache(nn.Module):
    def __init__(self, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int):
        super().__init__()
        # Initialize k and v as buffers to ensure they're part of the module state
        self.k = mx.zeros(
            (layers, bsz, max_seq_len, kv_heads, head_dim),
            dtype=mx.bfloat16
        )
        self.v = mx.zeros(
            (layers, bsz, max_seq_len, kv_heads, head_dim),
            dtype=mx.bfloat16
        )

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        """Creates a new KVCache instance with initialized k and v tensors."""
        return cls(layers, bsz, max_seq_len, kv_heads, head_dim)

    def update(
        self,
        xk: mx.array,
        xv: mx.array,
        layer_idx: int,
        cur_pos: int,
        n_rep: int
    ):
        """
        Updates the cache with new key and value tensors.

        Args:
            xk (mx.array): New key tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            xv (mx.array): New value tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            layer_idx (int): The index of the layer to update.
            cur_pos (int): The current position in the sequence to start inserting.
            n_rep (int): The number of times to repeat the keys and values along the sequence dimension.

        Returns:
            Tuple[mx.array, mx.array]:
                - keys: Updated or repeated keys tensor.
                - values: Updated or repeated values tensor.
        """
        # Ensure xk and xv have the correct device and dtype
        xk = xk.astype(self.k.dtype)
        xv = xv.astype(self.v.dtype)


        # Update the k and v tensors in the specified layer and position
        insert_len = xk.shape[1] 
        self.k[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xk
        self.v[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xv

        if cur_pos == 0:
            # If inserting at the beginning, repeat the new keys and values
            keys = mx.repeat(xk, n_rep, axis=2)
            values = mx.repeat(xv, n_rep, axis=2)
        else:
            # Otherwise, repeat the existing keys and values from the cache
            keys = mx.repeat(self.k[layer_idx], n_rep, axis=2)
            values = mx.repeat(self.v[layer_idx], n_rep, axis=2)

        return keys, values, self

    def clear(self):
        """Resets the k and v caches to zeros."""
        self.k = mx.zeros_like(self.k)
        self.v = mx.zeros_like(self.v)