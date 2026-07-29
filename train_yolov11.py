from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11DiSP_PFAE_CNN.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TriDom-YOLO on a YOLO-format detection dataset.")
    parser.add_argument("--data", required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--weights", help="Optional checkpoint used to initialize the model.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="TriDom-YOLO model YAML path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--name", default="tridom-yolo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        optimizer="SGD",
        lr0=0.001,
        lrf=0.01,
        box=8.0,
        dfl=2.0,
        patience=50,
        mosaic=0.8,
        mixup=0.2,
        seed=0,
        deterministic=True,
        project=args.project,
        name=args.name,
        save=True,
        val=True,
    )


if __name__ == "__main__":
    main()
