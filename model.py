from typing import Any
from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


class ResNetBlock(nn.Module):
    filters: int
    conv: Any
    norm: Any
    act: Any

    @nn.compact
    def __call__(self, x: chex.ArrayBatched):
        i = x
        x = self.conv(self.filters, 3)(x)
        x = self.norm()(x)
        x = self.act(x)
        x = self.conv(self.filters, 3)(x)
        x = self.norm(scale_init=nn.initializers.zeros_init())(x)
        return self.act(x + i)


class AZNet(nn.Module):
    num_actions: int
    filters: int = 64
    num_blocks: int = 6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(
            nn.Conv,
            use_bias=False,
            dtype=self.dtype,
        )
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            dtype=self.dtype,
        )
        act = nn.relu

        x = conv(self.filters, 3)(x.astype(self.dtype))
        x = norm()(x)
        x = act(x)
        for _ in range(self.num_blocks):
            x = ResNetBlock(self.filters, conv, norm, act)(x)

        l = conv(256, x.shape[1:3], padding="VALID")(x)
        l = norm()(l)
        l = act(l)
        l = conv(self.num_actions, 1, use_bias=True)(l)

        v = conv(self.filters, x.shape[1:3], padding="VALID")(x)
        v = norm()(v)
        v = act(v)
        v = conv(self.filters, 1)(v)
        v = norm()(v)
        v = act(v)
        v = conv(1, 1, use_bias=True)(v)
        v = jnp.tanh(v)

        return l.reshape(l.shape[0], -1).astype(jnp.float32), v.reshape(-1).astype(
            jnp.float32
        )


class TrainState(train_state.TrainState):
    batch_stats: Any


def create_train_state(
    rng: chex.PRNGKey,
    model: AZNet,
    inp_shape: chex.Shape,
    lr=1e-3,
) -> TrainState:
    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(rng, jnp.zeros(inp_shape))
    params, batch_stats = variables["params"], variables["batch_stats"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(lr),
        batch_stats=batch_stats,
    )
    return state


transforms = [
    lambda x: x,
    lambda x: jnp.rot90(x, k=1, axes=(0, 1)).reshape(x.shape),
    lambda x: jnp.rot90(x, k=2, axes=(0, 1)).reshape(x.shape),
    lambda x: jnp.rot90(x, k=3, axes=(0, 1)).reshape(x.shape),
    lambda x: jnp.flip(x, axis=0).reshape(x.shape),
    lambda x: jnp.flip(x, axis=1).reshape(x.shape),
    lambda x: jnp.transpose(x, axes=(1, 0, 2)).reshape(x.shape),
    lambda x: jnp.flip(jnp.transpose(x, axes=(1, 0, 2)), axis=0).reshape(x.shape),
]
inv_transforms = [
    lambda x: x,
    lambda x: jnp.rot90(x, k=-1, axes=(0, 1)).reshape(x.shape),
    lambda x: jnp.rot90(x, k=-2, axes=(0, 1)).reshape(x.shape),
    lambda x: jnp.rot90(x, k=-3, axes=(0, 1)).reshape(x.shape),
    lambda x: jnp.flip(x, axis=0).reshape(x.shape),
    lambda x: jnp.flip(x, axis=1).reshape(x.shape),
    lambda x: jnp.transpose(x, axes=(1, 0, 2)).reshape(x.shape),
    lambda x: jnp.transpose(jnp.flip(x, axis=0), axes=(1, 0, 2)).reshape(x.shape),
]


def model_evaluate(model: TrainState, obs: chex.Array, rng: chex.PRNGKey):
    index = jax.random.randint(rng, (obs.shape[0],), 0, len(transforms))
    obs = jax.vmap(jax.lax.switch, in_axes=(0, None, 0))(index, transforms, obs)
    logits, value = model.apply_fn(
        {"params": model.params, "batch_stats": model.batch_stats},
        obs,
        train=False,
        mutable=False,
    )
    logits_shape = logits.shape
    logits = logits.reshape(-1, obs.shape[1], obs.shape[2], 1)
    logits = jax.vmap(jax.lax.switch, in_axes=(0, None, 0))(
        index, inv_transforms, logits
    )
    logits = logits.reshape(*logits_shape)
    return logits, value


if __name__ == "__main__":
    import tqdm

    model = create_train_state(jax.random.PRNGKey(0), AZNet(15 * 15), (1, 15, 15, 3))
    obs = jnp.zeros((4096, 15, 15, 3))
    logits, value = model_evaluate(model, obs, jax.random.PRNGKey(0))

    for _ in tqdm.trange(1000000):
        logits, value = model_evaluate(model, obs, jax.random.PRNGKey(0))
