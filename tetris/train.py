"""PPO training for Tetris using Stable Baselines3."""

import argparse
import signal
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym

from .env import make_env

CHECKPOINT_DIR = Path("checkpoints")
TB_LOG_DIR = Path("tb_logs")


class TetrisCNN(BaseFeaturesExtractor):
    """Small CNN for the 20x10 Tetris board."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 20 * 10, features_dim),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tetris agent with PPO")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps (default: 2M)")
    parser.add_argument("--envs", type=int, default=8, help="Parallel environments (default: 8)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    TB_LOG_DIR.mkdir(exist_ok=True)

    # Create vectorized environments
    env = SubprocVecEnv([make_env() for _ in range(args.envs)])
    eval_env = DummyVecEnv([lambda: Monitor(make_env()())])

    policy_kwargs = {
        "features_extractor_class": TetrisCNN,
        "features_extractor_kwargs": {"features_dim": 256},
    }

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env, tensorboard_log=str(TB_LOG_DIR), device=device)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(TB_LOG_DIR),
            device=device,
        )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Network parameters: {total_params:,}")
    print(f"Training for {args.timesteps:,} timesteps with {args.envs} envs")

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // args.envs,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="tetris",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(CHECKPOINT_DIR),
        eval_freq=25_000 // args.envs,
        n_eval_episodes=5,
        deterministic=True,
    )

    def handle_interrupt(signum, frame):
        print("\n\nInterrupted! Saving model...")
        model.save(CHECKPOINT_DIR / "interrupted")
        print(f"Model saved to {CHECKPOINT_DIR / 'interrupted'}")
        env.close()
        eval_env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(CHECKPOINT_DIR / "final")
    print(f"Training complete. Model saved to {CHECKPOINT_DIR / 'final'}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
