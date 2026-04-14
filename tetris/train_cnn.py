"""DQN training loop for Tetris using CNN on full board state."""

import argparse
import copy
import random
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from .env import TetrisEngine
from .cnn_model import TetrisCNN

CHECKPOINT_DIR = Path("checkpoints_cnn")


def get_board_tensor(env: TetrisEngine) -> np.ndarray:
    """Get 20x10 binary board as (1, 20, 10) float32 array."""
    return env.get_board().astype(np.float32).reshape(1, 20, 10)


def simulate_placement(env: TetrisEngine, placement: dict) -> np.ndarray:
    """Simulate a placement and return the resulting board as tensor.
    Uses the feature computation's simulated board."""
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

    # Extract playfield and clear lines
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


def train(model, env, num_episodes, device):
    optimizer = Adam(model.parameters(), lr=1e-4)
    target_model = copy.deepcopy(model)
    target_model.eval()
    target_update_freq = 100

    replay_buffer = deque(maxlen=50_000)
    batch_size = 128
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.001

    best_lines = 0
    stats = []
    batch_start = time.time()

    for episode in range(1, num_episodes + 1):
        env.reset()
        total_reward = 0
        total_lines = 0
        pieces = 0

        while not env.done:
            placements = env.get_valid_placements()
            if not placements:
                break

            # Simulate all placements to get board states
            boards = []
            for p in placements:
                boards.append(simulate_placement(env, p))

            # Epsilon-greedy
            if random.random() < epsilon:
                chosen_idx = random.randrange(len(placements))
            else:
                board_tensor = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                with torch.no_grad():
                    scores = model(board_tensor).squeeze()
                chosen_idx = scores.argmax().item() if scores.dim() > 0 else 0

            chosen_board = boards[chosen_idx]
            reward, done, info = env.execute_placement(placements[chosen_idx])

            lines = info.get("lines_cleared", 0)
            shaped_reward = 1 + lines * 10

            # Hole penalty
            board = env.get_board()
            holes = 0
            for col in range(10):
                found = False
                for row in range(20):
                    if board[row, col] > 0:
                        found = True
                    elif found:
                        holes += 1

            total_reward += shaped_reward
            total_lines += lines
            pieces += 1

            # Get next state boards
            if not done:
                next_placements = env.get_valid_placements()
                next_boards = [simulate_placement(env, p) for p in next_placements] if next_placements else []
            else:
                next_boards = []

            replay_buffer.append({
                "board": chosen_board,
                "reward": shaped_reward,
                "next_boards": next_boards,
                "done": done,
            })

        # Train from replay buffer
        if len(replay_buffer) >= batch_size:
            batch = random.sample(list(replay_buffer), batch_size)

            states = torch.tensor(
                np.array([b["board"] for b in batch]), dtype=torch.float32
            ).to(device)

            targets = []
            for b in batch:
                if b["done"] or not b["next_boards"]:
                    targets.append(b["reward"])
                else:
                    next_t = torch.tensor(
                        np.array(b["next_boards"]), dtype=torch.float32
                    ).to(device)
                    with torch.no_grad():
                        next_scores = target_model(next_t).squeeze()
                        best_next = next_scores.max().item() if next_scores.dim() > 0 else next_scores.item()
                    targets.append(b["reward"] + gamma * best_next)

            targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)
            predictions = model(states)
            loss = torch.nn.functional.mse_loss(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        stats.append({"episode": episode, "pieces": pieces, "lines": total_lines, "reward": total_reward})

        if total_lines > best_lines:
            best_lines = total_lines
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")

        if episode % 50 == 0:
            elapsed = time.time() - batch_start
            recent = stats[-50:]
            avg_pieces = np.mean([s["pieces"] for s in recent])
            avg_lines = np.mean([s["lines"] for s in recent])
            avg_reward = np.mean([s["reward"] for s in recent])
            print(f"Episode {episode:5d} | "
                  f"Pieces: {avg_pieces:5.1f} | "
                  f"Lines: {avg_lines:5.2f} | "
                  f"Reward: {avg_reward:7.1f} | "
                  f"Best lines: {best_lines} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Time: {elapsed:.0f}s")
            torch.save(model.state_dict(), CHECKPOINT_DIR / "latest.pt")
            batch_start = time.time()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tetris agent with CNN DQN")
    parser.add_argument("--episodes", type=int, default=10000, help="Training episodes")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Starting epsilon")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = TetrisCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
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

    train(model, env, args.episodes, device)
    torch.save(model.state_dict(), CHECKPOINT_DIR / "final.pt")
    print(f"Training complete. Model saved to {CHECKPOINT_DIR / 'final.pt'}")


if __name__ == "__main__":
    main()
