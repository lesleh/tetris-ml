# Tetris ML

A neural network that plays Tetris, trained with deep reinforcement learning (DQN).

## How it works

For each piece, the agent enumerates every valid placement (column + rotation), computes 7 features of the resulting board state, and picks the placement with the highest predicted score.

### Features

| Feature | Description |
|---------|-------------|
| Lines cleared | Number of rows cleared by this placement |
| Holes | Empty cells with a filled cell above them |
| Bumpiness | Sum of height differences between adjacent columns |
| Total height | Sum of all column heights |
| Max height | Height of the tallest column |
| Height diff | Difference between tallest and shortest column |
| Height variance | Variance of column heights (catches lopsided stacking) |

### Reward function

- **+1** per piece placed
- **+lines * 10** for clearing lines (linear scaling)
- **-4** per new hole created

### Architecture

- **Model**: MLP with two hidden layers (128 neurons each)
- **Parameters**: ~17,500
- **Training**: DQN with target network, epsilon-greedy exploration, experience replay (100K buffer)
- **Input**: 7 board features
- **Output**: Placement score (scalar)
- **Exported model**: ~78KB ONNX

### Pre-trained model

Pre-trained weights are available in `models/`:
- `models/best_model.pt` — PyTorch weights
- `models/best_model.onnx` — ONNX for browser inference

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Train

```bash
train                                                          # Train from scratch (10000 episodes)
train --checkpoint checkpoints/best_model.pt --epsilon 0.01    # Resume from checkpoint
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

Best game: 9,236 lines cleared. Average: ~1,000+ lines per game.

The model plays clean Tetris with minimal holes and balanced stacking. It occasionally drifts into lopsided states after thousands of pieces, which is a fundamental limitation of feature-based evaluation (the model sees aggregate stats, not spatial layout).

Training takes a few hours on an M4 Pro MacBook. The breakthrough typically happens around episode 3,500-4,500 when epsilon drops below 0.03.
