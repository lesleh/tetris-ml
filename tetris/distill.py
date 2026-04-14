"""Distill MLP into CNN via imitation learning.

The CNN learns to pick the same placement as the MLP by training
on (all_boards, best_index) pairs — a classification problem.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .env import TetrisEngine
from .model import TetrisMLP
from .cnn_model import TetrisCNN
from . import board_sim

CHECKPOINT_DIR = Path("checkpoints_cnn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill MLP into CNN")
    parser.add_argument("--mlp-checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    use_c = board_sim.is_available()
    if use_c:
        print("Using C board simulation")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load MLP teacher
    mlp = TetrisMLP()
    mlp.load_state_dict(torch.load(args.mlp_checkpoint, map_location="cpu", weights_only=True))
    mlp.eval()
    print(f"Loaded MLP from {args.mlp_checkpoint}")

    # CNN student
    cnn = TetrisCNN().to(device)
    print(f"CNN parameters: {sum(p.numel() for p in cnn.parameters()):,}")

    optimizer = Adam(cnn.parameters(), lr=1e-3)

    interrupted = False
    def handle_interrupt(signum, frame):
        nonlocal interrupted
        if interrupted:
            sys.exit(1)
        interrupted = True
        print("\n\nInterrupted! Saving...")
        torch.save(cnn.state_dict(), CHECKPOINT_DIR / "distilled.pt")
        print(f"Saved to {CHECKPOINT_DIR / 'distilled.pt'}")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_interrupt)

    # Train online — play games with MLP, train CNN each step
    total_loss = 0
    total_correct = 0
    total_samples = 0
    batch_boards = []
    batch_labels = []

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()

        for g in range(args.games):
            env = TetrisEngine()
            env.reset()
            pieces = 0

            while not env.done and pieces < 1000:
                inner = env.env.unwrapped
                placements = env.get_valid_placements()
                if not placements:
                    break

                # MLP picks best placement
                features = torch.tensor([p["features"] for p in placements], dtype=torch.float32)
                with torch.no_grad():
                    mlp_scores = mlp(features).squeeze()
                best_idx = mlp_scores.argmax().item() if mlp_scores.dim() > 0 else 0

                # Get board states for CNN
                if use_c:
                    playfield = (inner.board[:20, 4:14] > 1).astype(np.int8)
                    piece_matrix = inner.active_tetromino.matrix.astype(np.uint8)
                    boards_arr, meta, count = board_sim.enumerate_all(playfield, piece_matrix)
                    if count == 0:
                        break
                    boards = boards_arr[:count].astype(np.float32).reshape(count, 1, 20, 10)
                else:
                    from .train_cnn import simulate_placement
                    boards = np.array([simulate_placement(env, p) for p in placements], dtype=np.float32)
                    count = len(placements)

                # Accumulate batch: all boards for this placement + which one MLP chose
                if best_idx < count:
                    batch_boards.append(boards[:count])
                    batch_labels.append(best_idx)

                # Train when we have enough
                if len(batch_boards) >= args.batch_size:
                    # CNN scores all boards for each sample, cross-entropy against MLP's choice
                    loss_sum = 0
                    correct = 0
                    for boards_set, label in zip(batch_boards, batch_labels):
                        boards_t = torch.tensor(boards_set, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            pass  # just need forward
                        scores = cnn(boards_t).squeeze()  # (num_placements,)
                        target = torch.tensor(label, dtype=torch.long).to(device)
                        loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
                        loss_sum += loss
                        if scores.argmax().item() == label:
                            correct += 1

                    avg_loss = loss_sum / len(batch_boards)
                    optimizer.zero_grad()
                    avg_loss.backward()
                    optimizer.step()

                    epoch_loss += avg_loss.item() * len(batch_boards)
                    epoch_correct += correct
                    epoch_total += len(batch_boards)

                    batch_boards = []
                    batch_labels = []

                env.execute_placement(placements[best_idx])
                pieces += 1

            if (g + 1) % 20 == 0:
                acc = epoch_correct / max(1, epoch_total) * 100
                print(f"  Epoch {epoch} | Game {g + 1}/{args.games} | "
                      f"Loss: {epoch_loss / max(1, epoch_total):.4f} | "
                      f"Accuracy: {acc:.1f}%")

        elapsed = time.time() - epoch_start
        acc = epoch_correct / max(1, epoch_total) * 100
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Loss: {epoch_loss / max(1, epoch_total):.4f} | "
              f"Accuracy: {acc:.1f}% | "
              f"Time: {elapsed:.0f}s")

        torch.save(cnn.state_dict(), CHECKPOINT_DIR / "distilled.pt")

    torch.save(cnn.state_dict(), CHECKPOINT_DIR / "distilled.pt")
    print(f"\nSaved to {CHECKPOINT_DIR / 'distilled.pt'}")
    print("Fine-tune with: train-cnn --checkpoint checkpoints_cnn/distilled.pt --epsilon 0.05")


if __name__ == "__main__":
    main()
