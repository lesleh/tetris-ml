"""Distill MLP knowledge into CNN by supervised learning on board evaluations."""

import argparse
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from .env import TetrisEngine
from .model import TetrisMLP
from .cnn_model import TetrisCNN
from . import board_sim

CHECKPOINT_DIR = Path("checkpoints_cnn")


def generate_training_data(mlp: TetrisMLP, num_games: int, device: str = "cpu") -> list[tuple[np.ndarray, float]]:
    """Play games with the MLP and record (board_state, mlp_score) pairs."""
    data = []
    env = TetrisEngine()
    use_c = board_sim.is_available()

    for g in range(num_games):
        env.reset()
        pieces = 0

        while not env.done and pieces < 2000:
            inner = env.env.unwrapped
            playfield = (inner.board[:20, 4:14] > 1).astype(np.int8)
            piece_matrix = inner.active_tetromino.matrix.astype(np.uint8)

            if use_c:
                boards_arr, meta, count = board_sim.enumerate_all(playfield, piece_matrix)
                if count == 0:
                    break
            else:
                placements = env.get_valid_placements()
                if not placements:
                    break
                count = len(placements)

            # Get MLP scores for all placements
            if use_c:
                # Need features for MLP — use env's feature computation
                placements = env.get_valid_placements()
                if not placements:
                    break

            features = torch.tensor(
                [p["features"] for p in placements], dtype=torch.float32
            )
            with torch.no_grad():
                mlp_scores = mlp(features).squeeze()

            # Get board states for each placement
            if use_c:
                boards = boards_arr[:count].astype(np.float32).reshape(count, 1, 20, 10)
            else:
                from .train_cnn import simulate_placement
                boards = np.array([simulate_placement(env, p) for p in placements], dtype=np.float32)

            # Record all (board, score) pairs
            for i in range(min(len(placements), count)):
                score = mlp_scores[i].item() if mlp_scores.dim() > 0 else mlp_scores.item()
                data.append((boards[i], score))

            # Play the best move
            best_idx = mlp_scores.argmax().item() if mlp_scores.dim() > 0 else 0
            env.execute_placement(placements[best_idx])
            pieces += 1

        lines = sum(1 for _ in range(20) if False)  # placeholder
        if (g + 1) % 5 == 0 or g == 0:
            print(f"  Game {g + 1}/{num_games} | {pieces} pieces | {len(data)} samples total")

    return data


def train_cnn_supervised(cnn: TetrisCNN, data: list, device: str, epochs: int = 50, batch_size: int = 256):
    """Train CNN to predict MLP scores from board states."""
    optimizer = Adam(cnn.parameters(), lr=1e-3)

    boards = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32).to(device)
    scores = torch.tensor([d[1] for d in data], dtype=torch.float32).unsqueeze(1).to(device)

    n = len(data)
    print(f"Training CNN on {n} samples for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        indices = torch.randperm(n)
        total_loss = 0
        batches = 0

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_boards = boards[batch_idx]
            batch_scores = scores[batch_idx]

            predictions = cnn(batch_boards)
            loss = torch.nn.functional.mse_loss(predictions, batch_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return avg_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill MLP into CNN")
    parser.add_argument("--mlp-checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="MLP checkpoint to distill from")
    parser.add_argument("--games", type=int, default=100, help="Games to generate training data from")
    parser.add_argument("--epochs", type=int, default=50, help="Supervised training epochs")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load MLP teacher
    mlp = TetrisMLP()
    mlp.load_state_dict(torch.load(args.mlp_checkpoint, map_location="cpu", weights_only=True))
    mlp.eval()
    print(f"Loaded MLP from {args.mlp_checkpoint}")

    # Generate training data (interruptible)
    data = []
    interrupted = False

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        if interrupted:
            sys.exit(1)
        interrupted = True
        print(f"\n\nInterrupted! Training on {len(data)} samples collected so far...")

    signal.signal(signal.SIGINT, handle_interrupt)

    print(f"\nGenerating training data from {args.games} games (ctrl+c to stop early and train)...")
    for g in range(args.games):
        if interrupted:
            break
        env = TetrisEngine()
        env.reset()
        pieces = 0
        use_c = board_sim.is_available()

        while not env.done and pieces < 2000:
            inner = env.env.unwrapped
            placements = env.get_valid_placements()
            if not placements:
                break

            features = torch.tensor([p["features"] for p in placements], dtype=torch.float32)
            with torch.no_grad():
                mlp_scores = mlp(features).squeeze()

            if use_c:
                playfield = (inner.board[:20, 4:14] > 1).astype(np.int8)
                piece_matrix = inner.active_tetromino.matrix.astype(np.uint8)
                boards_arr, meta, count = board_sim.enumerate_all(playfield, piece_matrix)
                boards = boards_arr[:count].astype(np.float32).reshape(count, 1, 20, 10)
            else:
                from .train_cnn import simulate_placement
                boards = np.array([simulate_placement(env, p) for p in placements], dtype=np.float32)
                count = len(placements)

            for i in range(min(len(placements), count)):
                score = mlp_scores[i].item() if mlp_scores.dim() > 0 else mlp_scores.item()
                data.append((boards[i], score))

            best_idx = mlp_scores.argmax().item() if mlp_scores.dim() > 0 else 0
            env.execute_placement(placements[best_idx])
            pieces += 1

        if (g + 1) % 5 == 0 or g == 0:
            print(f"  Game {g + 1}/{args.games} | {pieces} pieces | {len(data)} samples total")

    print(f"Total samples: {len(data)}")

    if len(data) < 1000:
        print("Not enough data to train. Run more games.")
        return

    # Train CNN
    cnn = TetrisCNN().to(device)
    print(f"\nCNN parameters: {sum(p.numel() for p in cnn.parameters()):,}")

    interrupted = False

    train_cnn_supervised(cnn, data, device, epochs=args.epochs)

    # Save
    torch.save(cnn.state_dict(), CHECKPOINT_DIR / "distilled.pt")
    print(f"\nSaved distilled CNN to {CHECKPOINT_DIR / 'distilled.pt'}")
    print("Now fine-tune with: train-cnn --checkpoint checkpoints_cnn/distilled.pt --epsilon 0.05")


if __name__ == "__main__":
    main()
