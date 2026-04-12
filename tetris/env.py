"""Wrapped Tetris environment with shaped rewards and CNN-friendly observations."""

import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris


class TetrisWrapper(gym.Wrapper):
    """Wraps tetris-gymnasium with:
    - Flat board observation (1, 20, 10) suitable for CNN
    - Shaped reward: lines cleared (squared), hole penalty, survival bonus, game over penalty
    """

    def __init__(self, render_mode=None):
        env = Tetris(render_mode=render_mode)
        super().__init__(env)

        # Override observation space to a simple 2D grid
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1, 20, 10), dtype=np.float32
        )

        self._prev_holes = 0
        self._prev_height_variance = 0.0

    def _extract_board(self, obs: dict) -> np.ndarray:
        """Extract the 20x10 playfield as a binary (1, 20, 10) array."""
        board = obs["board"]
        # Playfield is rows 0-19, cols 4-13 (inside the walls)
        playfield = board[:20, 4:14]
        # Binary: anything > 1 is a placed piece (1 = wall, 0 = empty)
        # Actually: 0 = empty, 1 = wall/border, 2-8 = piece types
        binary = (playfield > 1).astype(np.float32)
        return binary.reshape(1, 20, 10)

    def _count_holes(self, board: np.ndarray) -> int:
        """Count holes: empty cells with a filled cell anywhere above them."""
        holes = 0
        # board is (1, 20, 10), squeeze to (20, 10)
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
        """Get the height of each column (0 = empty, 20 = full)."""
        grid = board.squeeze()
        heights = np.zeros(10)
        for col in range(10):
            for row in range(20):
                if grid[row, col] > 0:
                    heights[col] = 20 - row
                    break
        return heights

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_holes = 0
        self._prev_height_variance = 0.0
        return self._extract_board(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        board = self._extract_board(obs)

        lines = info.get("lines_cleared", 0)

        # Shaped reward
        shaped_reward = 0.0

        # Lines cleared (big reward)
        if lines > 0:
            shaped_reward += lines * lines * 10

        # Reward for placing a piece (base env gives reward=1 on lock)
        if reward > 0:
            shaped_reward += 1.0

            # Flatness bonus: reward keeping the surface even (only on piece placement)
            heights = self._column_heights(board)
            variance = float(np.var(heights))
            # Reward decrease in variance (got flatter), penalize increase
            delta = self._prev_height_variance - variance
            shaped_reward += delta * 0.5
            self._prev_height_variance = variance

            # Hole penalty (gentle, only on placement)
            holes = self._count_holes(board)
            new_holes = holes - self._prev_holes
            if new_holes > 0:
                shaped_reward -= new_holes * 0.5
            self._prev_holes = holes

        # Game over penalty
        if terminated:
            shaped_reward -= 10.0

        return board, shaped_reward, terminated, truncated, info


def make_env(render_mode=None):
    """Factory function for creating wrapped Tetris environments."""
    def _init():
        return TetrisWrapper(render_mode=render_mode)
    return _init
