import os
import jax
import jax.numpy as jnp
import optax
import chex
import tqdm
from typing import NamedTuple
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import pgx

from model import AZNet, TrainState, create_train_state, model_evaluate
import mcts
import gomoku

env = gomoku.Env(3, 3)

seed = 42
weight_decay = 1e-5
num_batchsize_train = 4096
num_batchsize_test = 1024
num_batchsize_self_play = 1024
num_self_play = 2**17 // num_batchsize_self_play
num_max_step_self_play = 9
num_mcts_simulations = 32


class TrainData(NamedTuple):
    obs: chex.ArrayDevice
    probs: chex.ArrayDevice
    value: chex.ArrayDevice
    mask: chex.ArrayDevice


@jax.jit
def self_play(model: TrainState, key: chex.PRNGKey) -> TrainData:
    key, subkey = jax.random.split(key)
    state = jax.vmap(env.init)(jax.random.split(subkey, num_batchsize_self_play))

    def body(s: pgx.State, k: chex.PRNGKey):
        policy = mcts.mcts_search(
            (env, model),
            s,
            model_evaluate,
            k,
            num_simulations=num_mcts_simulations,
            max_depth=num_max_step_self_play,
        )
        ns = jax.vmap(env.step)(s, policy.action)
        return ns, (
            jax.vmap(env.observe)(s),
            policy.action_weights,
            s.terminated,
            s.current_player,
            ns.rewards,
        )

    key, subkey = jax.random.split(key)
    _, (obs, probs, terminated, players, rewards) = jax.lax.scan(
        body, state, jax.random.split(subkey, num_max_step_self_play)
    )
    value = jax.vmap(lambda x, i: x[i], in_axes=(0, 1), out_axes=1)(
        jnp.sum(rewards, axis=0), players
    )

    mask = ~terminated.reshape(-1)
    obs = obs.reshape(-1, *obs.shape[2:])
    probs = probs.reshape(-1, *probs.shape[2:])
    value = value.reshape(-1, *value.shape[2:])
    return TrainData(obs, probs, value, mask)


@jax.jit
def train_step(model: TrainState, batch):

    def loss_fn(params):
        (logits, value), new_model = model.apply_fn(
            {"params": params, "batch_stats": model.batch_stats},
            batch.obs,
            train=True,
            mutable=["batch_stats"],
        )
        mask = batch.mask / jnp.sum(batch.mask)

        policy_loss = optax.losses.safe_softmax_cross_entropy(logits, batch.probs)
        policy_loss = (policy_loss * mask).sum()

        value_loss = optax.losses.l2_loss(value, batch.value)
        value_loss = (value_loss * mask).sum()

        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
        weight_penalty = weight_decay * 0.5 * weight_l2

        return policy_loss + value_loss + weight_penalty, (
            new_model,
            policy_loss,
            value_loss,
            weight_penalty,
        )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(model.params)
    new_model, policy_loss, value_loss, wd_loss = aux[1]
    new_model = model.apply_gradients(
        grads=grads,
        batch_stats=new_model["batch_stats"],
    )
    return new_model, policy_loss, value_loss, wd_loss


@jax.jit
def eval_step(model: TrainState, key: chex.PRNGKey):
    k0, k1, k2 = jax.random.split(key, 3)
    model_player = jax.random.randint(k0, (num_batchsize_test,), 0, env.num_players)
    state = jax.vmap(env.init)(jax.random.split(k1, num_batchsize_test))

    def cond(val):
        s, k, r = val
        return ~s.terminated.all()

    def body(val):
        s, k, r = val
        k, k0, k1 = jax.random.split(k, 3)
        action_model = mcts.mcts_search(
            (env, model),
            state,
            model_evaluate,
            k0,
            num_simulations=num_mcts_simulations,
            dirichlet_fraction=0.0,
            temperature=0.0,
        ).action
        action_random = mcts.mcts_search(
            (env, model),
            state,
            mcts.random_evaluate,
            k1,
            num_simulations=num_mcts_simulations,
            dirichlet_fraction=0.0,
            temperature=0.0,
        ).action
        action = jnp.where(
            s.current_player == model_player,
            action_model,
            action_random,
        )
        s = jax.vmap(env.step)(s, action)
        r = r + s.rewards
        return s, k, r

    state, key, rewards = jax.lax.while_loop(cond, body, (state, k2, state.rewards))
    rewards = jax.vmap(lambda x, i: x[i])(rewards, model_player)
    return (rewards[..., None] == jnp.float32([1.0, 0.0, -1.0])).sum(
        0
    ) / num_batchsize_test


@jax.jit
def fast_eval_step(model: TrainState, key: chex.PRNGKey):
    k0, k1, k2 = jax.random.split(key, 3)
    model_player = jax.random.randint(k0, (num_batchsize_test,), 0, env.num_players)
    state = jax.vmap(env.init)(jax.random.split(k1, num_batchsize_test))

    def cond(val):
        s, k, r = val
        return ~s.terminated.all()

    def body(val):
        s, k, r = val
        k, k0, k1 = jax.random.split(k, 3)
        logits_model = model_evaluate((env, model), s, k0)[0]
        logits_random = jnp.log(s.legal_action_mask)
        logits = jnp.where(
            jnp.expand_dims(s.current_player == model_player, -1),
            logits_model,
            logits_random,
        )
        action = jax.random.categorical(k1, logits)
        s = jax.vmap(env.step)(s, action)
        r = r + s.rewards
        return s, k, r

    state, key, rewards = jax.lax.while_loop(cond, body, (state, k2, state.rewards))
    rewards = jax.vmap(lambda x, i: x[i])(rewards, model_player)
    return (rewards[..., None] == jnp.float32([1.0, 0.0, -1.0])).sum(
        0
    ) / num_batchsize_test


def main():
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = create_train_state(
        subkey,
        AZNet(env.num_actions, 64, 6, dtype=jnp.bfloat16),
        (1, *env.observation_shape),
    )
    if os.path.exists("./checkpoint"):
        model = restore_checkpoint(os.path.abspath("./checkpoint"), model)
    key, k0 = jax.random.split(key)
    rate = fast_eval_step(model, k0) * 100
    print(
        f"step {model.step}, win={rate[0].item():4.2f}, draw={rate[1].item():4.2f}, loss={rate[2].item():4.2f}"
    )

    while model.step < 2**18:
        key, k0, k1, k2 = jax.random.split(key, 4)

        data = [
            self_play(model, k)
            for k in tqdm.tqdm(jax.random.split(k0, num_self_play), desc="self-play")
        ]
        data = jax.tree.map(lambda *x: jnp.concat(x), *data)

        ixs = jax.random.permutation(k1, data.obs.shape[0])
        ixs = jnp.split(ixs, ixs.shape[0] // num_batchsize_train)
        for ids in tqdm.tqdm(ixs, desc="train"):
            model, ploss, vloss, wdloss = train_step(
                model,
                jax.tree_util.tree_map(lambda x: x[ids], data),
            )

        rate = fast_eval_step(model, k2) * 100
        print(
            f"step {model.step}, ploss={ploss.item():.5f}, vloss={vloss.item():.5f}, wdloss={wdloss.item():.5f}, win={rate[0].item():4.2f}, draw={rate[1].item():4.2f}, loss={rate[2].item():4.2f}"
        )
        save_checkpoint(
            os.path.abspath("./checkpoint"),
            model,
            int(model.step),
            keep=5,
        )


if __name__ == "__main__":
    main()
