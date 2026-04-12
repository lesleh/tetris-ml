"""Export trained Tetris MLP to ONNX."""

import argparse
from pathlib import Path

import torch

from tetris.model import TetrisMLP


def main():
    parser = argparse.ArgumentParser(description="Export Tetris model to ONNX")
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("-o", "--output", type=str, default="tetris-model.onnx")
    args = parser.parse_args()

    model = TetrisMLP()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, 4)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    size = Path(args.output).stat().st_size
    print(f"Exported to {args.output} ({size} bytes)")


if __name__ == "__main__":
    main()
