"""Watch a trained Tetris agent play."""

import argparse
import os
import time
from pathlib import Path

from stable_baselines3 import PPO

from .env import TetrisWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch trained Tetris agent")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_model",
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", "-n", type=int, default=5, help="Number of episodes")
    parser.add_argument("--delay", "-d", type=float, default=0.1, help="Delay between steps (seconds)")
    parser.add_argument("--mode", "-m", choices=["human", "ansi"], default="human",
                        help="Render mode (default: human/pygame)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() and not Path(str(ckpt) + ".zip").exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Train first: train")
        return

    model = PPO.load(args.checkpoint)
    env = TetrisWrapper(render_mode=args.mode)

    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            steps += 1

            if args.mode == "ansi":
                os.system("clear")
                print(env.render())
                print(f"Step: {steps}  Reward: {total_reward:.1f}")

            time.sleep(args.delay)

            if terminated or truncated:
                print(f"Episode {ep + 1}: {steps} steps, reward: {total_reward:.1f}")
                break

    env.close()


if __name__ == "__main__":
    main()
