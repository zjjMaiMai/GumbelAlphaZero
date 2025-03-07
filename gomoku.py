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


def to_svg(states: State, filename: str):
    import svgwrite
    from dataclasses import dataclass

    @dataclass
    class ColorSet:
        p1_color: str = "black"
        p2_color: str = "white"
        p1_outline: str = "black"
        p2_outline: str = "black"
        background_color: str = "white"
        grid_color: str = "black"
        text_color: str = "black"

    config = {
        "GRID_SIZE": -1,
        "BOARD_WIDTH": -1,
        "BOARD_HEIGHT": -1,
        "COLOR_THEME": "light",
        "COLOR_SET": ColorSet(),
        "SCALE": 1.0,
    }

    def _get_nth_state(states: State, i):
        return jax.tree_util.tree_map(lambda x: x[i], states)

    def _make_gomoku_dwg(dwg, state, config):
        GRID_SIZE = config["GRID_SIZE"]
        BOARD_SIZE = config["BOARD_WIDTH"]
        color_set = config["COLOR_SET"]

        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (BOARD_SIZE * GRID_SIZE, BOARD_SIZE * GRID_SIZE),
                # stroke=svgwrite.rgb(10, 10, 16, "%"),
                fill=color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        hlines = board_g.add(dwg.g(id="hlines", stroke=color_set.grid_color))
        for y in range(1, BOARD_SIZE - 1):
            hlines.add(
                dwg.line(
                    start=(0, GRID_SIZE * y),
                    end=(
                        GRID_SIZE * (BOARD_SIZE - 1),
                        GRID_SIZE * y,
                    ),
                    stroke_width="0.5px",
                )
            )
        vlines = board_g.add(dwg.g(id="vline", stroke=color_set.grid_color))
        for x in range(1, BOARD_SIZE - 1):
            vlines.add(
                dwg.line(
                    start=(GRID_SIZE * x, 0),
                    end=(
                        GRID_SIZE * x,
                        GRID_SIZE * (BOARD_SIZE - 1),
                    ),
                    stroke_width="0.5px",
                )
            )
        board_g.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_SIZE - 1) * GRID_SIZE,
                    (BOARD_SIZE - 1) * GRID_SIZE,
                ),
                fill="none",
                stroke=color_set.grid_color,
                stroke_width="2px",
            )
        )

        # stones
        board = jnp.clip(state.board, -1, 1)
        for xy, stone in enumerate(board):
            if stone == -1:
                continue
            stone_y = xy // BOARD_SIZE * GRID_SIZE
            stone_x = xy % BOARD_SIZE * GRID_SIZE

            color = color_set.p1_color if stone == 0 else color_set.p2_color
            outline = color_set.p1_outline if stone == 0 else color_set.p2_outline
            board_g.add(
                dwg.circle(
                    center=(stone_x, stone_y),
                    r=GRID_SIZE / 2.2,
                    stroke=outline,
                    fill=color,
                )
            )
            # if xy == state.last_action:
            #     board_g.add(
            #         dwg.circle(
            #             center=(stone_x, stone_y),
            #             r=GRID_SIZE / 2.2 / 3,
            #             fill="red",
            #         )
            #     )

        board_g.translate(GRID_SIZE / 2, GRID_SIZE / 2)
        return board_g

    try:
        SIZE = len(states.current_player)
        WIDTH = math.ceil(math.sqrt(SIZE - 0.1))
        if SIZE - (WIDTH - 1) ** 2 >= WIDTH:
            HEIGHT = WIDTH
        else:
            HEIGHT = WIDTH - 1
        if SIZE == 1:
            states = _get_nth_state(states, 0)
    except TypeError:
        SIZE = 1
        WIDTH = 1
        HEIGHT = 1

    config["GRID_SIZE"] = 25
    config["BOARD_WIDTH"] = int(math.sqrt(states.board.shape[-1]))  # type: ignore
    config["BOARD_HEIGHT"] = int(math.sqrt(states.board.shape[-1]))  # type: ignore

    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    SCALE = config["SCALE"]

    dwg = svgwrite.Drawing(
        filename,
        (
            (BOARD_WIDTH + 1) * GRID_SIZE * WIDTH * SCALE,
            (BOARD_HEIGHT + 1) * GRID_SIZE * HEIGHT * SCALE,
        ),
    )
    group = dwg.g()

    # background
    group.add(
        dwg.rect(
            (0, 0),
            (
                (BOARD_WIDTH + 1) * GRID_SIZE * WIDTH,
                (BOARD_HEIGHT + 1) * GRID_SIZE * HEIGHT,
            ),
            fill=config["COLOR_SET"].background_color,
        )
    )

    if SIZE == 1:
        g = _make_gomoku_dwg(dwg, states, config)
        g.translate(
            GRID_SIZE * 1 / 2,
            GRID_SIZE * 1 / 2,
        )
        group.add(g)
        group.scale(SCALE)
        dwg.add(group)
        return dwg.save()

    for i in range(SIZE):
        x = i % WIDTH
        y = i // WIDTH
        _state = _get_nth_state(states, i)
        g = _make_gomoku_dwg(dwg, _state, config)

        g.translate(
            GRID_SIZE * 1 / 2 + (BOARD_WIDTH + 1) * GRID_SIZE * x,
            GRID_SIZE * 1 / 2 + (BOARD_HEIGHT + 1) * GRID_SIZE * y,
        )
        group.add(g)
        group.add(
            dwg.rect(
                (
                    (BOARD_WIDTH + 1) * GRID_SIZE * x,
                    (BOARD_HEIGHT + 1) * GRID_SIZE * y,
                ),
                (
                    (BOARD_WIDTH + 1) * GRID_SIZE,
                    (BOARD_HEIGHT + 1) * GRID_SIZE,
                ),
                fill="none",
                stroke="gray",
            )
        )
    group.scale(SCALE)
    dwg.add(group)
    return dwg.save()
