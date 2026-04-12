"""Watch a trained Tetris agent play with Pygame rendering."""

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO

from .env import TetrisWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch trained Tetris agent")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_model",
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", "-n", type=int, default=5, help="Number of episodes")
    parser.add_argument("--delay", "-d", type=float, default=0.05, help="Delay between steps (seconds)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() and not Path(str(ckpt) + ".zip").exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Train first: train")
        return

    model = PPO.load(args.checkpoint)
    env = TetrisWrapper(render_mode="human")

    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(args.delay)

            if terminated or truncated:
                print(f"Episode {ep + 1}: {steps} steps, reward: {total_reward:.1f}")
                break

    env.close()


if __name__ == "__main__":
    main()
