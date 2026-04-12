"""Watch a trained Tetris agent play."""

import argparse
import os
import time
from pathlib import Path

import torch

from .env import TetrisEngine
from .model import TetrisMLP


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch trained Tetris agent")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", "-n", type=int, default=5, help="Number of episodes")
    parser.add_argument("--delay", "-d", type=float, default=0.3, help="Delay between pieces (seconds)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Train first: train")
        return

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

            features = torch.tensor(
                [p["features"] for p in placements], dtype=torch.float32
            )
            with torch.no_grad():
                scores = model(features).squeeze()
            best_idx = scores.argmax().item() if scores.dim() > 0 else 0
            chosen = placements[best_idx]

            reward, done, info = env.execute_placement(chosen)
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
