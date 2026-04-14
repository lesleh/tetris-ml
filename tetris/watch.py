"""Watch a trained Tetris agent play."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from .env import TetrisEngine
from .model import TetrisMLP
from .cnn_model import TetrisCNN


def simulate_placement_board(env: TetrisEngine, placement: dict) -> np.ndarray:
    """Simulate a placement and return resulting board as (1, 20, 10)."""
    inner = env.env.unwrapped
    tetromino = inner.active_tetromino
    for _ in range(placement["rotation"]):
        tetromino = inner.rotate(tetromino, True)

    matrix = tetromino.matrix
    h, w = matrix.shape

    board_copy = inner.board.copy()
    for r in range(h):
        for c in range(w):
            if matrix[r, c] != 0:
                board_copy[placement["y"] + r, placement["x"] + c] = matrix[r, c]

    playfield = board_copy[:20, 4:14]
    binary = (playfield > 1).astype(np.float32)

    cleaned = []
    for row in range(20):
        if not np.all(binary[row] > 0):
            cleaned.append(binary[row])
    result = np.zeros((20, 10), dtype=np.float32)
    offset = 20 - len(cleaned)
    for i, row in enumerate(cleaned):
        result[offset + i] = row

    return result.reshape(1, 20, 10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch trained Tetris agent")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--cnn", action="store_true", help="Use CNN model")
    parser.add_argument("--episodes", "-n", type=int, default=5, help="Number of episodes")
    parser.add_argument("--delay", "-d", type=float, default=0.3, help="Delay between pieces (seconds)")
    args = parser.parse_args()

    # Default checkpoint based on model type
    if args.checkpoint is None:
        args.checkpoint = "checkpoints_cnn/best_model.pt" if args.cnn else "checkpoints/best_model.pt"

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Train first: train" + ("-cnn" if args.cnn else ""))
        return

    if args.cnn:
        model = TetrisCNN()
    else:
        model = TetrisMLP()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    env = TetrisEngine()

    for ep in range(args.episodes):
        env.reset()
        total_reward = 0
        total_lines = 0
        pieces = 0

        while not env.done:
            os.system("clear")
            print(env.render_board())
            print(f"\nPiece: {pieces + 1}  Lines: {total_lines}  Reward: {total_reward:.0f}")

            placements = env.get_valid_placements()
            if not placements:
                break

            if args.cnn:
                boards = np.array([simulate_placement_board(env, p) for p in placements])
                with torch.no_grad():
                    scores = model(torch.tensor(boards, dtype=torch.float32)).squeeze()
            else:
                features = torch.tensor(
                    [p["features"] for p in placements], dtype=torch.float32
                )
                with torch.no_grad():
                    scores = model(features).squeeze()

            best_idx = scores.argmax().item() if scores.dim() > 0 else 0

            reward, done, info = env.execute_placement(placements[best_idx])
            total_reward += reward
            total_lines += info.get("lines_cleared", 0)
            pieces += 1

            time.sleep(args.delay)

        os.system("clear")
        print(env.render_board())
        print(f"\nGame Over! Pieces: {pieces}  Lines: {total_lines}  Reward: {total_reward:.0f}")
        time.sleep(2)

    print(f"\nDone watching {args.episodes} episodes.")


if __name__ == "__main__":
    main()
