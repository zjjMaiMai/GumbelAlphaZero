import os
import jax
import jax.numpy as jnp
import chex
import pgx
import mctx
from flax.training.checkpoints import restore_checkpoint

from model import AZNet, create_train_state, model_evaluate
import pgx
import gomoku


def batch(state: pgx.State):
    return jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)


def unbatch(state: pgx.State):
    return jax.tree.map(lambda x: jnp.squeeze(x, 0), state)


def recurrent_fn(
    model,
    key: chex.PRNGKey,
    action: chex.ArrayBatched,
    state: pgx.State,
):
    player = state.current_player
    state = jax.vmap(env.step)(state, action)

    logits, value = model_evaluate(model, state.observation, key)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = jax.vmap(lambda x, i: x[i])(state.rewards, player)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(state.terminated, 0.0, -1.0),
        prior_logits=logits,
        value=jnp.where(state.terminated, 0.0, value),
    )
    return recurrent_fn_output, state


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    env = gomoku.Env(9, 5)
    key, subkey = jax.random.split(key)
    state = env.init(subkey)
    env.to_svg(state, "/tmp/game.svg")

    key, subkey = jax.random.split(key)
    model = create_train_state(
        subkey,
        AZNet(env.num_actions),
        (1, *env.observation_shape),
    )
    elo = jnp.float32(600)
    model, elo = restore_checkpoint(os.path.abspath("./checkpoint"), (model, elo))
    print(f"elo: {elo.item()}")

    while not state.terminated.all():
        y, x = map(
            int, input("Now state: /tmp/game.svg, please input action:").split(",")
        )
        action = y * env.size + x
        state = env.step(state, action)
        env.to_svg(state, "/tmp/game.svg")

        key, k0, k1 = jax.random.split(key, 3)
        state = batch(state)
        logits, value = model_evaluate(model, state.observation, k0)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy = mctx.gumbel_muzero_policy(
            model,
            k1,
            root,
            recurrent_fn,
            128,
            invalid_actions=~state.legal_action_mask,
            gumbel_scale=0.0,
        )
        state = unbatch(state)
        state = env.step(state, policy.action[0])
        print(
            "value: ",
            policy.search_tree.raw_values[0, policy.search_tree.ROOT_INDEX],
        )
        env.to_svg(state, "/tmp/game.svg")
