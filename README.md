# Tetris ML

A neural network that plays Tetris, trained with deep reinforcement learning (DQN).

## How it works

For each piece, the agent enumerates every valid placement (column + rotation), computes 5 features of the resulting board state, and picks the placement with the highest predicted score.

### Features

| Feature | Description |
|---------|-------------|
| Lines cleared | Number of rows cleared by this placement |
| Holes | Empty cells with a filled cell above them |
| Bumpiness | Sum of height differences between adjacent columns |
| Total height | Sum of all column heights |
| Max height | Height of the tallest column |

### Architecture

- **Model**: MLP with two hidden layers (128 neurons each)
- **Parameters**: ~17,000
- **Training**: DQN with target network, epsilon-greedy exploration, experience replay
- **Input**: 5 board features
- **Output**: Placement score (scalar)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Train

```bash
train                                    # Train from scratch (5000 episodes)
train --checkpoint checkpoints/best_model.pt --epsilon 0.05  # Resume
```

### Watch

```bash
watch                     # Watch AI play in terminal with colored blocks
watch --delay 0.1         # Faster playback
```

### Evaluate

```bash
evaluate -n 100           # Run 100 games, print stats
```

### Export to ONNX

```bash
python scripts/export_onnx.py            # Export best model
```

## Results

Best game: 2,102 lines cleared. Average: ~600+ lines per game after training.
