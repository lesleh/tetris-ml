"""Tetris environment wrapper with placement enumeration and feature extraction."""

import numpy as np
from tetris_gymnasium.envs.tetris import Tetris


class TetrisEngine:
    """Wraps tetris-gymnasium. Enumerates valid placements and computes features."""

    def __init__(self, render_mode=None):
        self.env = Tetris(render_mode=render_mode)
        self._done = False

    def reset(self):
        self.env.reset()
        self._done = False

    @property
    def done(self):
        return self._done

    def get_board(self) -> np.ndarray:
        """Return the 20x10 playfield (placed pieces only, no active piece)."""
        inner = self.env.unwrapped
        board = inner.board[:20, 4:14]
        return (board > 1).astype(np.int8)

    def get_valid_placements(self) -> list[dict]:
        """Enumerate all valid placements for the current piece.

        Returns list of dicts with keys:
            rotation, x, y, features (lines_cleared, holes, bumpiness, total_height)
        """
        inner = self.env.unwrapped
        placements = []
        seen = set()

        tetromino = inner.active_tetromino
        for rot in range(4):
            if rot > 0:
                tetromino = inner.rotate(tetromino, True)

            matrix = tetromino.matrix
            h, w = matrix.shape

            for x in range(inner.width + inner.padding * 2 - w + 1):
                # Check if piece can exist at top
                if inner.collision(tetromino, x, 0):
                    continue

                # Drop to lowest valid y
                y = 0
                while not inner.collision(tetromino, x, y + 1):
                    y += 1

                # Simulate placement on board copy
                board_copy = inner.board.copy()
                for r in range(h):
                    for c in range(w):
                        if matrix[r, c] != 0:
                            board_copy[y + r, x + c] = matrix[r, c]

                # Extract playfield
                playfield = board_copy[:20, 4:14]
                binary = (playfield > 1).astype(np.int8)

                # Deduplicate by board state
                board_key = binary.tobytes()
                if board_key in seen:
                    continue
                seen.add(board_key)

                # Count cleared lines
                lines = 0
                cleaned = []
                for row in range(20):
                    if np.all(binary[row] > 0):
                        lines += 1
                    else:
                        cleaned.append(binary[row])
                if lines > 0:
                    binary = np.zeros((20, 10), dtype=np.int8)
                    offset = 20 - len(cleaned)
                    for i, row in enumerate(cleaned):
                        binary[offset + i] = row

                features = self._compute_features(binary, lines)
                placements.append({
                    "rotation": rot,
                    "x": x,
                    "y": y,
                    "features": features,
                })

        return placements

    def _compute_features(self, board: np.ndarray, lines_cleared: int) -> list[float]:
        """Compute [lines_cleared, holes, bumpiness, total_height, max_height]."""
        heights = np.zeros(10)
        for col in range(10):
            for row in range(20):
                if board[row, col] > 0:
                    heights[col] = 20 - row
                    break

        holes = 0
        for col in range(10):
            found = False
            for row in range(20):
                if board[row, col] > 0:
                    found = True
                elif found:
                    holes += 1

        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(9))
        total_height = sum(heights)
        max_height = max(heights)

        height_diff = max(heights) - min(heights)
        return [float(lines_cleared), float(holes), float(bumpiness), float(total_height), float(max_height), float(height_diff)]

    def execute_placement(self, placement: dict) -> tuple[float, bool, dict]:
        """Execute a placement by rotating, moving, and hard dropping."""
        inner = self.env.unwrapped
        actions = inner.actions

        # Rotate
        for _ in range(placement["rotation"]):
            self.env.step(actions.rotate_clockwise)

        # Move to target x
        current_x = inner.x
        target_x = placement["x"]
        if target_x < current_x:
            for _ in range(current_x - target_x):
                self.env.step(actions.move_left)
        elif target_x > current_x:
            for _ in range(target_x - current_x):
                self.env.step(actions.move_right)

        # Hard drop
        obs, reward, terminated, truncated, info = self.env.step(actions.hard_drop)
        self._done = terminated or truncated

        lines = info.get("lines_cleared", 0)
        shaped_reward = 1 + lines ** 2 * 10

        if self._done:
            # Check if death was caused by a well (large height difference)
            board = self.get_board()
            heights = np.zeros(10)
            for col in range(10):
                for row in range(20):
                    if board[row, col] > 0:
                        heights[col] = 20 - row
                        break
            well_depth = max(heights) - min(heights)
            if well_depth >= 4:
                shaped_reward -= 200  # Died with a well — heavily penalize
            else:
                shaped_reward -= 20   # Normal death

        return shaped_reward, self._done, info

    def render_board(self) -> str:
        """Render board with ANSI colors."""
        inner = self.env.unwrapped
        obs = inner._get_obs()
        board = obs["board"]

        COLORS = {
            0: "\033[90m", 2: "\033[36m", 3: "\033[34m", 4: "\033[33m",
            5: "\033[93m", 6: "\033[32m", 7: "\033[35m", 8: "\033[31m",
        }
        RESET = "\033[0m"
        BLOCK = "\u2588\u2588"
        EMPTY = "\u2591\u2591"

        lines = []
        for r in range(20):
            row = ""
            for c in range(4, 14):
                val = board[r, c]
                if val > 1:
                    row += f"{COLORS.get(int(val), COLORS[0])}{BLOCK}{RESET}"
                else:
                    row += f"{COLORS[0]}{EMPTY}{RESET}"
            lines.append(row)
        return "\n".join(lines)
