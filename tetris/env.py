"""Wrapped Tetris environment with simplified action space and shaped rewards."""

import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris


class TetrisWrapper(gym.Wrapper):
    """Wraps tetris-gymnasium with:
    - Simplified action space: 40 actions = 10 columns x 4 rotations
    - Board observation (1, 20, 10) suitable for CNN
    - Shaped rewards
    """

    # 40 actions: col 0-9, rotation 0-3
    # action = col * 4 + rotation
    NUM_ACTIONS = 40

    def __init__(self, render_mode=None):
        env = Tetris(render_mode=render_mode)
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1, 20, 10), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.NUM_ACTIONS)

        self._prev_holes = 0

    def _extract_board(self, obs: dict) -> np.ndarray:
        """Extract the 20x10 playfield as a binary (1, 20, 10) array, excluding active piece."""
        board = obs["board"]
        mask = obs["active_tetromino_mask"]
        playfield = board[:20, 4:14]
        active = mask[:20, 4:14]
        # Placed pieces: board > 1 AND not the active piece
        binary = ((playfield > 1) & (active == 0)).astype(np.float32)
        return binary.reshape(1, 20, 10)

    def _count_holes(self, board: np.ndarray) -> int:
        """Count holes: empty cells with a filled cell anywhere above them."""
        holes = 0
        grid = board.squeeze()
        for col in range(10):
            found_block = False
            for row in range(20):
                if grid[row, col] > 0:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def _column_heights(self, board: np.ndarray) -> np.ndarray:
        """Get the height of each column."""
        grid = board.squeeze()
        heights = np.zeros(10)
        for col in range(10):
            for row in range(20):
                if grid[row, col] > 0:
                    heights[col] = 20 - row
                    break
        return heights

    def _execute_placement(self, target_col: int, rotation: int) -> tuple:
        """Rotate piece and move to target column, then hard drop."""
        inner = self.env.unwrapped
        actions = inner.actions

        # Apply rotations
        for _ in range(rotation):
            obs, _, term, trunc, info = self.env.step(actions.rotate_clockwise)
            if term or trunc:
                return obs, 0, term, trunc, info

        # Figure out current column (x position relative to playfield)
        current_col = inner.x - 4

        # Move to target column
        moves = target_col - current_col
        move_action = actions.move_right if moves > 0 else actions.move_left
        for _ in range(abs(moves)):
            obs, _, term, trunc, info = self.env.step(move_action)
            if term or trunc:
                return obs, 0, term, trunc, info

        # Hard drop
        return self.env.step(actions.hard_drop)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_holes = 0
        return self._extract_board(obs), info

    def step(self, action):
        target_col = action // 4
        rotation = action % 4

        obs, reward, terminated, truncated, info = self._execute_placement(target_col, rotation)
        board = self._extract_board(obs)

        lines = info.get("lines_cleared", 0)

        # Shaped reward
        shaped_reward = 0.0

        # Lines cleared (big reward)
        if lines > 0:
            shaped_reward += lines * lines * 100

        # Placement bonus
        shaped_reward += 1.0

        # Flatness bonus
        heights = self._column_heights(board)
        variance = float(np.var(heights))
        flatness_bonus = max(0, 5.0 - variance * 0.5)
        shaped_reward += flatness_bonus

        # Hole penalty
        holes = self._count_holes(board)
        new_holes = holes - self._prev_holes
        if new_holes > 0:
            shaped_reward -= new_holes * 0.5
        self._prev_holes = holes

        # Game over penalty
        if terminated:
            shaped_reward -= 50.0

        return board, shaped_reward, terminated, truncated, info


def make_env(render_mode=None):
    """Factory function for creating wrapped Tetris environments."""
    def _init():
        return TetrisWrapper(render_mode=render_mode)
    return _init
