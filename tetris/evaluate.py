"""Evaluate a trained Tetris agent over many episodes."""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .env import TetrisWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tetris agent")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_model")
    parser.add_argument("--episodes", "-n", type=int, default=100)
    args = parser.parse_args()

    model = PPO.load(args.checkpoint)
    env = TetrisWrapper()

    rewards = []
    steps_list = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                rewards.append(total_reward)
                steps_list.append(steps)
                if (ep + 1) % 10 == 0:
                    print(f"Episode {ep + 1}/{args.episodes}: {steps} steps, reward: {total_reward:.1f}")
                break

    env.close()

    print(f"\nResults ({args.episodes} episodes):")
    print(f"  Reward: mean={np.mean(rewards):.1f}, median={np.median(rewards):.1f}, max={np.max(rewards):.1f}")
    print(f"  Steps:  mean={np.mean(steps_list):.0f}, median={np.median(steps_list):.0f}, max={np.max(steps_list)}")


if __name__ == "__main__":
    main()
