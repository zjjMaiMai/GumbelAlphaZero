import os
import jax
import jax.numpy as jnp
import pgx
from flax.training.checkpoints import restore_checkpoint

from model import AZNet, create_train_state, model_evaluate
import pgx
import gomoku
import mcts


def batch(state: pgx.State):
    return jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)


def unbatch(state: pgx.State):
    return jax.tree.map(lambda x: jnp.squeeze(x, 0), state)


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    env = gomoku.Env(3, 3)
    key, subkey = jax.random.split(key)
    state = env.init(subkey)

    key, subkey = jax.random.split(key)
    model = create_train_state(
        subkey,
        AZNet(env.num_actions, 64, 6, dtype=jnp.bfloat16),
        (1, *env.observation_shape),
    )
    model = restore_checkpoint(os.path.abspath("./checkpoint"), model)

    while not state.terminated:
        key, subkey = jax.random.split(key)
        policy = mcts.mcts_search(
            (env, model),
            batch(state),
            model_evaluate,
            subkey,
            num_simulations=32,
            dirichlet_fraction=0.0,
            temperature=0.0,
        )
        state = env.step(state, policy.action[0])
        gomoku.to_svg(state, "game.svg")

        action = int(input("Now state: ./game.svg, please input action:"))
        state = env.step(state, action)
