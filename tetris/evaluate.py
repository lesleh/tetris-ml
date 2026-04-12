"""Evaluate a trained Tetris agent over many episodes."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .env import TetrisEngine
from .model import TetrisMLP


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tetris agent")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--episodes", "-n", type=int, default=100)
    args = parser.parse_args()

    model = TetrisMLP()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    env = TetrisEngine()
    lines_list = []
    pieces_list = []

    for ep in range(args.episodes):
        env.reset()
        total_lines = 0
        pieces = 0

        while not env.done:
            placements = env.get_valid_placements()
            if not placements:
                break

            features = torch.tensor(
                [p["features"] for p in placements], dtype=torch.float32
            )
            with torch.no_grad():
                scores = model(features).squeeze()
            best_idx = scores.argmax().item() if scores.dim() > 0 else 0

            _, done, info = env.execute_placement(placements[best_idx])
            total_lines += info.get("lines_cleared", 0)
            pieces += 1

        lines_list.append(total_lines)
        pieces_list.append(pieces)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{args.episodes}: {pieces} pieces, {total_lines} lines")

    print(f"\nResults ({args.episodes} episodes):")
    print(f"  Lines:  mean={np.mean(lines_list):.1f}, median={np.median(lines_list):.0f}, max={np.max(lines_list)}")
    print(f"  Pieces: mean={np.mean(pieces_list):.0f}, median={np.median(pieces_list):.0f}, max={np.max(pieces_list)}")


if __name__ == "__main__":
    main()
