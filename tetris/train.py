"""DQN training loop for Tetris with feature-based placement scoring."""

import argparse
import copy
import random
import signal
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from .env import TetrisEngine
from .model import TetrisMLP

CHECKPOINT_DIR = Path("checkpoints")


def train(model, env, num_episodes, device, start_epsilon=1.0):
    optimizer = Adam(model.parameters(), lr=5e-4)
    target_model = copy.deepcopy(model)
    target_model.eval()
    target_update_freq = 100  # update target network every N episodes

    replay_buffer = deque(maxlen=100_000)
    batch_size = 512
    gamma = 0.95
    epsilon = start_epsilon
    epsilon_decay = 0.998
    epsilon_min = 0.001

    best_lines = 0
    stats = []

    for episode in range(1, num_episodes + 1):
        env.reset()
        total_reward = 0
        total_lines = 0
        pieces = 0

        while not env.done:
            placements = env.get_valid_placements()
            if not placements:
                break

            # Epsilon-greedy
            if random.random() < epsilon:
                chosen = random.choice(placements)
            else:
                features = torch.tensor(
                    [p["features"] for p in placements], dtype=torch.float32
                ).to(device)
                with torch.no_grad():
                    scores = model(features).squeeze()
                best_idx = scores.argmax().item() if scores.dim() > 0 else 0
                chosen = placements[best_idx]

            reward, done, info = env.execute_placement(chosen)
            total_reward += reward
            total_lines += info.get("lines_cleared", 0)
            pieces += 1

            # Get next state placements
            next_placements = env.get_valid_placements() if not done else []

            replay_buffer.append({
                "features": chosen["features"],
                "reward": reward,
                "next_features": [p["features"] for p in next_placements],
                "done": done,
            })

        # Train from replay buffer
        if len(replay_buffer) >= batch_size:
            batch = random.sample(list(replay_buffer), batch_size)

            states = torch.tensor(
                [b["features"] for b in batch], dtype=torch.float32
            ).to(device)

            # Compute targets using target network (stable)
            targets = []
            for b in batch:
                if b["done"] or not b["next_features"]:
                    targets.append(b["reward"])
                else:
                    next_f = torch.tensor(b["next_features"], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        next_scores = target_model(next_f).squeeze()
                        best_next = next_scores.max().item() if next_scores.dim() > 0 else next_scores.item()
                    targets.append(b["reward"] + gamma * best_next)

            targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)
            predictions = model(states)
            loss = torch.nn.functional.mse_loss(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network periodically
        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        stats.append({"episode": episode, "pieces": pieces, "lines": total_lines, "reward": total_reward})

        if total_lines > best_lines:
            best_lines = total_lines
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")

        if episode % 50 == 0:
            recent = stats[-50:]
            avg_pieces = np.mean([s["pieces"] for s in recent])
            avg_lines = np.mean([s["lines"] for s in recent])
            avg_reward = np.mean([s["reward"] for s in recent])
            print(f"Episode {episode:5d} | "
                  f"Pieces: {avg_pieces:5.1f} | "
                  f"Lines: {avg_lines:5.2f} | "
                  f"Reward: {avg_reward:7.1f} | "
                  f"Best lines: {best_lines} | "
                  f"Epsilon: {epsilon:.3f}")
            torch.save(model.state_dict(), CHECKPOINT_DIR / "latest.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tetris agent with DQN")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes (default: 5000)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Starting epsilon (default: 1.0)")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    device = torch.device("cpu")

    model = TetrisMLP().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Network parameters: {total_params:,}")

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {args.checkpoint}")

    env = TetrisEngine()

    def handle_interrupt(signum, frame):
        print("\n\nInterrupted! Saving model...")
        torch.save(model.state_dict(), CHECKPOINT_DIR / "interrupted.pt")
        print(f"Saved to {CHECKPOINT_DIR / 'interrupted.pt'}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    train(model, env, args.episodes, device, start_epsilon=args.epsilon)
    torch.save(model.state_dict(), CHECKPOINT_DIR / "final.pt")
    print(f"Training complete. Model saved to {CHECKPOINT_DIR / 'final.pt'}")


if __name__ == "__main__":
    main()
