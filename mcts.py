import jax
import jax.numpy as jnp
import pgx
import chex
import mctx
import pygraphviz
from functools import partial
from typing import Optional, Callable, Any, Tuple, Sequence

from model import TrainState

Params = Tuple[pgx.Env, Optional[TrainState]]
EvaluateFn = Callable[
    [Params, pgx.State, chex.PRNGKey],
    Tuple[chex.ArrayBatched, chex.ArrayBatched],
]


def random_evaluate(
    params: Params,
    state: pgx.State,
    key: chex.PRNGKey,
    num_rollout: int = 128,
):
    env, _ = params

    def cond(val: Tuple[pgx.State, chex.PRNGKey]):
        s, r, _ = val
        return ~(s.terminated | s.truncated).all()

    def body(val: Tuple[pgx.State, chex.PRNGKey]):
        s, r, k = val
        k, sk = jax.random.split(k)
        logits = jnp.log(s.legal_action_mask.astype(jnp.float32))
        action = jax.random.categorical(sk, logits=logits)
        s = jax.vmap(env.step)(s, action)
        r = r + s.rewards
        return s, r, k

    def rollout(state, key):
        return jax.lax.while_loop(cond, body, (state, state.rewards, key))[1]

    value = jax.vmap(rollout, (None, 0))(
        state,
        jax.random.split(key, num_rollout),
    ).mean(0)
    logits = jnp.log(state.legal_action_mask)
    return logits, jax.vmap(lambda x, i: x[i])(value, state.current_player)


def recurrent_fn(
    params: Params,
    key: chex.PRNGKey,
    action: chex.ArrayBatched,
    state: pgx.State,
    evaluate_fn: EvaluateFn,
):
    env, _ = params
    player = state.current_player
    state = jax.vmap(env.step)(state, action)

    logits, value = evaluate_fn(params, state, key)
    reward = jax.vmap(lambda x, i: x[i])(state.rewards, player)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(state.terminated, 0, -1),
        prior_logits=logits,
        value=jnp.where(state.terminated, 0, value),
    )
    return recurrent_fn_output, state


def mcts_search(
    params: Params,
    state: pgx.State,
    evaluate_fn: EvaluateFn,
    key: chex.PRNGKey,
    num_simulations: int,
    **kwargs,
):
    key0, key1 = jax.random.split(key)
    logits, value = evaluate_fn(params, state, key0)
    root = mctx.RootFnOutput(
        prior_logits=logits,
        value=value,
        embedding=state,
    )
    return mctx.muzero_policy(
        params,
        rng_key=key1,
        root=root,
        recurrent_fn=partial(recurrent_fn, evaluate_fn=evaluate_fn),
        num_simulations=num_simulations,
        **kwargs,
    )


def convert_tree_to_graph(
    tree: mctx.Tree, action_labels: Optional[Sequence[str]] = None, batch_index: int = 0
) -> pygraphviz.AGraph:
    """Converts a search tree into a Graphviz graph.

    Args:
      tree: A `Tree` containing a batch of search data.
      action_labels: Optional labels for edges, defaults to the action index.
      batch_index: Index of the batch element to plot.

    Returns:
      A Graphviz graph representation of `tree`.
    """
    chex.assert_rank(tree.node_values, 2)
    batch_size = tree.node_values.shape[0]
    if action_labels is None:
        action_labels = range(tree.num_actions)
    elif len(action_labels) != tree.num_actions:
        raise ValueError(
            f"action_labels {action_labels} has the wrong number of actions "
            f"({len(action_labels)}). "
            f"Expecting {tree.num_actions}."
        )

    def node_to_str(node_i, reward=0, discount=1):
        return (
            f"{node_i}\n"
            f"Reward: {reward:.2f}\n"
            f"Discount: {discount:.2f}\n"
            f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"Visits: {tree.node_visits[batch_index, node_i]}\n"
        )

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        return (
            f"{action_labels[a_i]}\n"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"  # pytype: disable=unsupported-operands  # always-use-return-annotations
            f"p: {probs[a_i]:.2f}\n"
        )

    graph = pygraphviz.AGraph(directed=True)

    # Add root
    graph.add_node(0, label=node_to_str(node_i=0), color="green")
    # Add all other nodes and connect them up.
    for node_i in range(tree.num_simulations):
        for a_i in range(tree.num_actions):
            # Index of children, or -1 if not expanded
            children_i = tree.children_index[batch_index, node_i, a_i]
            if children_i >= 0:
                graph.add_node(
                    children_i,
                    label=node_to_str(
                        node_i=children_i,
                        reward=tree.children_rewards[batch_index, node_i, a_i],
                        discount=tree.children_discounts[batch_index, node_i, a_i],
                    ),
                    color="red",
                )
                graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

    return graph
