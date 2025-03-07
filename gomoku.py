import jax
import jax.numpy as jnp
import chex
import math

import pgx.core as core
from pgx._src.struct import dataclass


@dataclass
class State(core.State):
    board: chex.Array

    @property
    def env_id(self) -> core.EnvId:
        return "gomoku"


class Env(core.Env):
    def __init__(self, size, k):
        super().__init__()
        self.size = size
        self.k = k

    def _init(self, key: chex.PRNGKey) -> State:
        return State(
            board=jnp.full(self.size * self.size, jnp.int32(-1)),
            current_player=jnp.int32(0),
            observation=jnp.zeros((self.size * self.size, 2), dtype=jnp.bool_),
            rewards=jnp.float32([0.0, 0.0]),
            terminated=jnp.bool(False),
            truncated=jnp.bool(False),
            legal_action_mask=jnp.ones(self.size * self.size, dtype=jnp.bool_),
            _step_count=jnp.int32(0),
        )  # type:ignore

    def _step(self, state: State, action: chex.Array, key) -> State:
        board = state.board.at[action].set(state.current_player)
        yx = jnp.int32(jnp.divmod(action, self.size))
        dyx = jnp.int32([[1, 0], [0, 1], [1, 1], [1, -1]])
        steps = jnp.arange(-self.k + 1, self.k)
        nyx = jnp.expand_dims(steps, (0, 2)) * jnp.expand_dims(dyx, 1) + yx
        on_board = ((nyx >= 0) & (nyx < self.size)).all(-1)
        nyx = jnp.clip(nyx, 0, self.size)
        is_same = (
            board[nyx[..., 0] * self.size + nyx[..., 1]] == state.current_player
        ) & on_board
        window = jax.vmap(
            lambda i: jax.lax.dynamic_slice(is_same, (0, i), (is_same.shape[0], self.k))
        )(jnp.arange(self.k))
        is_win = window.all(-1).any()
        rewards = jnp.where(
            is_win,
            jnp.float32([-1, -1]).at[state.current_player].set(1),
            jnp.float32([0, 0]),
        )
        legal_action_mask = board < 0
        terminated = is_win | ~legal_action_mask.any()
        return state.replace(  # type:ignore
            board=board,
            current_player=(state.current_player + 1) % 2,
            rewards=rewards,
            terminated=terminated,
            legal_action_mask=legal_action_mask,
        )

    def _observe(self, state: State, player_id: chex.Array) -> chex.Array:
        return jnp.stack(
            [
                state.board == player_id,
                state.board == (player_id + 1) % 2,
                state.board == -1,
            ],
            axis=-1,
            dtype=jnp.float32,
        ).reshape(self.size, self.size, -1)

    @property
    def id(self) -> core.EnvId:
        return "gomoku"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2

    def to_svg(self, state, filename, grid_size=25, scale=1.0):
        import svgwrite

        grid_size = 25
        board_size = int(math.sqrt(state.board.shape[-1]))
        dwg = svgwrite.Drawing(
            filename, ((board_size + 1) * grid_size, (board_size + 1) * grid_size)
        )

        dwg.add(
            dwg.rect(
                (0, 0),
                ((board_size + 1) * grid_size, (board_size + 1) * grid_size),
                fill="white",
            )
        )

        for i in range(board_size):
            dwg.add(
                dwg.line(
                    (grid_size, grid_size * (i + 1)),
                    (board_size * grid_size, grid_size * (i + 1)),
                    stroke="black",
                    stroke_width="0.5px",
                )
            )
            dwg.add(
                dwg.text(
                    text=f"{i}",
                    insert=(grid_size // 2, grid_size * (i + 1)),
                    font_size="16px",
                    font_family="Menlo",
                    text_anchor="middle",
                    dominant_baseline="middle",
                )
            )
            dwg.add(
                dwg.line(
                    (grid_size * (i + 1), grid_size),
                    (grid_size * (i + 1), board_size * grid_size),
                    stroke="black",
                    stroke_width="0.5px",
                )
            )
            dwg.add(
                dwg.text(
                    text=f"{i}",
                    insert=(grid_size * (i + 1), grid_size // 2),
                    font_size="16px",
                    font_family="Menlo",
                    text_anchor="middle",
                    dominant_baseline="middle",
                )
            )

        dwg.add(
            dwg.rect(
                (grid_size, grid_size),
                (
                    board_size * grid_size - grid_size,
                    board_size * grid_size - grid_size,
                ),
                fill="none",
                stroke="black",
                stroke_width="2px",
            )
        )

        for idx, stone in enumerate(jnp.clip(state.board, -1, 1)):
            if stone != -1:
                y, x = divmod(idx, board_size)
                dwg.add(
                    dwg.circle(
                        center=((x + 1) * grid_size, (y + 1) * grid_size),
                        r=grid_size / 2.2,
                        stroke="black",
                        fill="black" if stone == 0 else "white",
                    )
                )

        dwg.save()
        return dwg
