import jax.numpy as jnp
import flax.nnx as nnx

class PosEmbedding(nnx.Module):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concat((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


if __name__ == "__main__":

    pe = PosEmbedding(8)

    x = jnp.arange(0, 16, dtype=jnp.int16)
    print(x)

    y = pe(x)
    print(y.shape)
    print(y)