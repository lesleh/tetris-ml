"""Watch a trained Tetris agent play."""

import argparse
import os
import time
from pathlib import Path

from stable_baselines3 import PPO

from .env import TetrisWrapper

# ANSI color codes for Tetris pieces
PIECE_COLORS = {
    0: "\033[90m",   # empty - dark gray
    1: "\033[37m",   # wall - white
    2: "\033[36m",   # I - cyan
    3: "\033[34m",   # J - blue
    4: "\033[33m",   # L - orange/yellow
    5: "\033[93m",   # O - bright yellow
    6: "\033[32m",   # S - green
    7: "\033[35m",   # T - purple
    8: "\033[31m",   # Z - red
}
RESET = "\033[0m"
BLOCK = "\u2588\u2588"  # full block, doubled for square aspect ratio


def render_board(env) -> str:
    """Render the board with colored blocks."""
    inner = env.env.unwrapped
    obs = inner._get_obs()
    board = obs["board"]
    mask = obs["active_tetromino_mask"]

    lines = []
    # Render playfield (rows 0-19, cols 4-13)
    for r in range(20):
        row = ""
        for c in range(4, 14):
            val = board[r, c]
            active = mask[r, c]
            if active > 0:
                color = PIECE_COLORS.get(int(active), PIECE_COLORS[7])
                row += f"{color}{BLOCK}{RESET}"
            elif val > 1:
                color = PIECE_COLORS.get(int(val), PIECE_COLORS[0])
                row += f"{color}{BLOCK}{RESET}"
            else:
                row += f"{PIECE_COLORS[0]}\u2591\u2591{RESET}"
        lines.append(row)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch trained Tetris agent")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_model",
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", "-n", type=int, default=5, help="Number of episodes")
    parser.add_argument("--delay", "-d", type=float, default=0.3, help="Delay between steps (seconds)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() and not Path(str(ckpt) + ".zip").exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Train first: train")
        return

    model = PPO.load(args.checkpoint)
    env = TetrisWrapper(render_mode=None)

    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            os.system("clear")
            print(render_board(env))
            print(f"\nPiece: {steps + 1}  Reward: {total_reward:.1f}")

            action, _ = model.predict(obs, deterministic=True)
            col = action // 4
            rot = action % 4
            print(f"Action: col={col} rot={rot}")

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            time.sleep(args.delay)

            if terminated or truncated:
                os.system("clear")
                print(render_board(env))
                print(f"\nGame Over! Pieces: {steps}  Reward: {total_reward:.1f}")
                time.sleep(1.5)
                break

    env.close()


if __name__ == "__main__":
    main()
