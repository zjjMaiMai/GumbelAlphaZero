import os
import jax
import jax.numpy as jnp
import optax
import chex
from typing import NamedTuple
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import pgx
from pgx.experimental import auto_reset

from model import AZNet, TrainState, create_train_state, model_evaluate, transforms
import mctx
import gomoku

env = gomoku.Env(9, 5)

seed = 42
num_batchsize_train = 1024
num_batchsize_test = 1024
num_batchsize_selfplay = 1024
num_step_selfplay = 256
num_simulations = 32


def recurrent_fn(
    model: TrainState,
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


class Trajectory(NamedTuple):
    obs: chex.Array
    prob: chex.Array
    reward: chex.Array
    discount: chex.Array
    terminated: chex.Array


class Sample(NamedTuple):
    obs: chex.Array
    prob: chex.Array
    value: chex.Array
    mask: chex.Array


@jax.jit
def self_play(model: TrainState, key: chex.PRNGKey) -> Sample:
    # we need to define a function that takes a state and a key and returns a new state and a trajectory
    def body(state: pgx.State, key: chex.PRNGKey) -> Trajectory:
        k0, k1, k2 = jax.random.split(key, 3)
        obs = state.observation
        logits, value = model_evaluate(model, obs, k0)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy = mctx.gumbel_muzero_policy(
            model,
            k1,
            root,
            recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
        )
        player = state.current_player
        state = jax.vmap(auto_reset(env.step, env.init))(
            state, policy.action, jax.random.split(k2, num_batchsize_selfplay)
        )
        return state, Trajectory(
            obs=obs,
            prob=policy.action_weights,
            reward=jax.vmap(lambda x, i: x[i])(state.rewards, player),
            discount=jnp.where(state.terminated, 0.0, -1.0),
            terminated=state.terminated,
        )

    k0, k1, k2 = jax.random.split(key, 3)
    state = jax.vmap(env.init)(jax.random.split(k0, num_batchsize_selfplay))
    _, traj = jax.lax.scan(
        body,
        state,
        jax.random.split(k1, num_step_selfplay),
    )

    # we calculate the value target
    def body(value: chex.ArrayBatched, traj: Trajectory):
        value = traj.reward + traj.discount * value
        return value, value

    _, value = jax.lax.scan(
        body,
        jnp.zeros_like(traj.reward[0]),
        traj,
        reverse=True,
    )
    mask = jnp.flip(jnp.cumsum(jnp.flip(traj.terminated, 0), 0), 0) >= 1
    sample = Sample(
        obs=traj.obs,
        prob=traj.prob,
        value=value,
        mask=mask,
    )
    # convert (T, B, ...) to (T * B, ...)
    sample = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), sample)

    # data augmentation
    def body(p, x):
        p = jax.tree_util.keystr(p, simple=True, separator="/")
        if p == "obs":
            return jnp.concat([jax.vmap(t)(x) for t in transforms])
        elif p == "prob":
            shape = x.shape
            x = x.reshape(-1, env.size, env.size, 1)
            x = jnp.concat([jax.vmap(t)(x) for t in transforms])
            x = x.reshape(-1, *shape[1:])
            return x
        else:
            return jnp.concat([x for _ in transforms])

    sample = jax.tree.map_with_path(body, sample)

    # shuffle sample and convert to minibatch
    ixs = jax.random.permutation(k2, sample.obs.shape[0])
    sample = jax.tree.map(lambda x: x[ixs], sample)
    sample = jax.tree.map(
        lambda x: x.reshape(-1, num_batchsize_train, *x.shape[1:]), sample
    )
    return sample


@jax.jit
def train_step(model: TrainState, batch: Sample):

    def loss_fn(params):
        (logits, value), new_model = model.apply_fn(
            {"params": params, "batch_stats": model.batch_stats},
            batch.obs,
            train=True,
            mutable=["batch_stats"],
        )
        policy_loss = optax.losses.safe_softmax_cross_entropy(logits, batch.prob).mean()
        value_loss = (optax.losses.l2_loss(value, batch.value) * batch.mask).mean()
        return policy_loss + value_loss, (
            new_model["batch_stats"],
            policy_loss,
            value_loss,
        )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(model.params)
    batch_stats, policy_loss, value_loss = aux[1]
    new_model = model.apply_gradients(
        grads=grads,
        batch_stats=batch_stats,
    )
    return new_model, policy_loss, value_loss


@jax.jit
def eval_step(
    model: TrainState,
    old_model: TrainState,
    old_elo: chex.Scalar,
    key: chex.PRNGKey,
):
    k0, k1, k2 = jax.random.split(key, 3)
    model_player = jax.random.randint(k0, (num_batchsize_test,), 0, env.num_players)
    state = jax.vmap(env.init)(jax.random.split(k1, num_batchsize_test))

    def cond(val):
        s, k, r = val
        return ~s.terminated.all()

    def body(val):
        s, k, r = val
        k, k0, k1, k2 = jax.random.split(k, 4)
        logits = jnp.where(
            jnp.expand_dims(s.current_player == model_player, -1),
            model_evaluate(model, s.observation, k0)[0],
            model_evaluate(old_model, s.observation, k1)[0],
        )
        logits = jnp.where(s.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        action = jax.random.categorical(k2, logits)
        s = jax.vmap(env.step)(s, action)
        r = r + s.rewards
        return s, k, r

    state, key, rewards = jax.lax.while_loop(cond, body, (state, k2, state.rewards))
    rewards = jax.vmap(lambda x, i: x[i])(rewards, model_player) * 0.5 + 0.5

    def body(elo, result):
        expected_score = 1 / (1 + 10 ** ((old_elo - elo) / 400))
        elo = elo + 32 * (result - expected_score)
        return elo, elo

    elo, _ = jax.lax.scan(body, old_elo, rewards)
    return elo


def main():
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = create_train_state(
        subkey,
        AZNet(env.num_actions),
        (num_batchsize_train, *env.observation_shape),
    )
    elo = jnp.float32(600)

    if os.path.exists("./checkpoint"):
        model, elo = restore_checkpoint(os.path.abspath("./checkpoint"), (model, elo))
    while True:
        key, k0, k1 = jax.random.split(key, 3)
        data = self_play(model, k0)

        old_model = model
        for ids in range(data.obs.shape[0]):
            model, ploss, vloss = train_step(
                model,
                jax.tree_util.tree_map(lambda x: x[ids], data),
            )
        elo = eval_step(model, old_model, elo, k1)
        print(
            f"step {model.step}"
            f", ploss={ploss.item():.5f}"
            f", vloss={vloss.item():.5f}"
            f", elo={elo.item():.2f}"
        )
        save_checkpoint(
            os.path.abspath("./checkpoint"),
            (model, elo),
            elo.item(),
            keep=5,
        )


if __name__ == "__main__":
    main()
