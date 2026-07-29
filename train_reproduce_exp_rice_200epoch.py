from __future__ import annotations

import argparse
from pathlib import Path

from prepare_rice1_exp_repro_dataset import TARGET_DATASET_YAML, prepare_dataset, print_stats
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_CFG = ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11n.yaml"
PRETRAINED_WEIGHTS = ROOT / "yolo11n.pt"
REFERENCE_WEIGHTS = ROOT / "best1.pt"


def validate_reference_weights(data_yaml: Path, device: str | int) -> None:
    metrics = YOLO(str(REFERENCE_WEIGHTS)).val(
        data=str(data_yaml),
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        plots=False,
    )
    print(f"reference best1.pt on normalized val set: mAP50={metrics.box.map50:.6f}, mAP50-95={metrics.box.map:.6f}")


def train(
    data_yaml: Path,
    device: str | int,
    workers: int,
    epochs: int,
    batch: int,
    run_name: str,
) -> None:
    model = YOLO(str(MODEL_CFG)).load(str(PRETRAINED_WEIGHTS))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=100,
        batch=batch,
        imgsz=640,
        optimizer="SGD",
        workers=workers,
        seed=0,
        deterministic=True,
        close_mosaic=0,
        lr0=0.01,
        lrf=0.01,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        project=str(ROOT / "runs" / "train"),
        name=run_name,
        save=True,
        val=True,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the 8/exp-rice-200epoch experiment as closely as possible.")
    parser.add_argument("--device", default="0", help="Training device passed to Ultralytics, for example 0 or cpu.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs. Defaults to the original run."
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size. Defaults to the original run.")
    parser.add_argument("--name", default="exp-rice-200epoch-repro", help="Run name under YOLOv11/runs/train.")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare the normalized detect dataset.")
    parser.add_argument(
        "--validate-reference",
        action="store_true",
        help="Validate best1.pt on the normalized dataset before training.",
    )
    args = parser.parse_args()

    data_yaml, stats = prepare_dataset()
    print_stats(stats)
    print(f"Using normalized dataset yaml: {TARGET_DATASET_YAML}")

    if args.validate_reference:
        validate_reference_weights(data_yaml, args.device)

    if args.prepare_only:
        return

    train(data_yaml, args.device, args.workers, args.epochs, args.batch, args.name)


if __name__ == "__main__":
    main()
