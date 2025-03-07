from typing import Any
from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

import pgx


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
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            dtype=self.dtype,
        )
        act = nn.leaky_relu

        x = conv(self.filters, 3)(x.astype(self.dtype))
        x = norm()(x)
        x = act(x)
        for _ in range(self.num_blocks):
            x = ResNetBlock(self.filters, conv, norm, act)(x)
        x = jnp.mean(x, axis=(1, 2), keepdims=True)

        l = conv(256, 1)(x)
        l = norm()(l)
        l = act(l)
        l = conv(self.num_actions, 1)(l)
        l = jnp.squeeze(l, (1, 2))

        v = conv(32, 1)(x)
        v = norm()(v)
        v = act(v)
        v = conv(1, 1)(v)
        v = jnp.tanh(v)
        v = jnp.squeeze(v, (1, 2, 3))
        return l.astype(jnp.float32), v.astype(jnp.float32)


class TrainState(train_state.TrainState):
    batch_stats: Any


def create_train_state(
    rng: chex.PRNGKey,
    model: AZNet,
    inp_shape: chex.Shape,
    lr=1e-2,
    warmup_step=1000,
    max_step=2**18,
) -> TrainState:
    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(rng, jnp.zeros(inp_shape))
    params, batch_stats = variables["params"], variables["batch_stats"]

    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=warmup_step,
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=max_step - warmup_step,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_step],
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(schedule_fn),
        batch_stats=batch_stats,
    )
    return state


def model_evaluate(params, state: pgx.State, key: chex.PRNGKey = None):
    env_, model = params
    logits, value = model.apply_fn(
        {"params": model.params, "batch_stats": model.batch_stats},
        jax.vmap(env_.observe)(state),
        train=False,
        mutable=False,
    )
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    return logits, value
