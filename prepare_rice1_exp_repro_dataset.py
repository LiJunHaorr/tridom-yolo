from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
SOURCE_DATASET_ROOT = WORKSPACE_ROOT / "8" / "rice_dataset" / "datasets"
TARGET_DATASET_ROOT = WORKSPACE_ROOT / "8" / "rice_dataset" / "datasets_detect_boxonly"
TARGET_DATASET_YAML = ROOT / "ultralytics" / "cfg" / "datasets" / "rice1_exp_repro_boxdetect.yaml"
SPLITS = ("train", "val", "test")
CLASS_NAMES = [
    "Bact_L_Blight",
    "Brn_Spot",
    "Healthy",
    "Leaf_Blast",
    "Scald",
    "Narrow_Br_Spot",
    "Neck_Blast",
    "Hispa",
]


@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    detect_lines: int = 0
    segment_lines: int = 0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_box(values: list[float]) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = (clamp01(v) for v in values[:4])
    return x_center, y_center, width, height


def polygon_to_box(values: list[float]) -> tuple[float, float, float, float]:
    xs = [clamp01(v) for v in values[0::2]]
    ys = [clamp01(v) for v in values[1::2]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_center = clamp01((min_x + max_x) / 2.0)
    y_center = clamp01((min_y + max_y) / 2.0)
    width = clamp01(max_x - min_x)
    height = clamp01(max_y - min_y)
    return x_center, y_center, width, height


def normalize_label_line(line: str, label_path: Path, line_no: int) -> tuple[str, bool] | None:
    raw = line.strip()
    if not raw:
        return None

    parts = raw.split()
    cls_token = parts[0]
    values = [float(v) for v in parts[1:]]

    if len(parts) == 5:
        x_center, y_center, width, height = normalize_box(values)
        return f"{cls_token} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}", False

    if len(values) < 4 or len(values) % 2 != 0:
        raise ValueError(f"Invalid label format in {label_path}:{line_no}: {raw}")

    x_center, y_center, width, height = polygon_to_box(values)
    return f"{cls_token} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}", True


def write_dataset_yaml(target_root: Path = TARGET_DATASET_ROOT, target_yaml: Path = TARGET_DATASET_YAML) -> Path:
    lines = [
        f"path: {target_root.as_posix()}",
        "",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    target_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_yaml


def prepare_dataset(
    source_root: Path = SOURCE_DATASET_ROOT,
    target_root: Path = TARGET_DATASET_ROOT,
    target_yaml: Path = TARGET_DATASET_YAML,
) -> tuple[Path, dict[str, SplitStats]]:
    stats: dict[str, SplitStats] = {}
    target_root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_stats = SplitStats()
        source_images = source_root / split / "images"
        source_labels = source_root / split / "labels"
        target_images = target_root / split / "images"
        target_labels = target_root / split / "labels"

        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(source_images.iterdir()):
            if not image_path.is_file():
                continue

            target_image_path = target_images / image_path.name
            if not target_image_path.exists() or target_image_path.stat().st_size != image_path.stat().st_size:
                shutil.copy2(image_path, target_image_path)
            split_stats.images += 1

        for label_path in sorted(source_labels.glob("*.txt")):
            normalized_lines: list[str] = []
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                normalized = normalize_label_line(line, label_path, line_no)
                if normalized is None:
                    continue

                normalized_line, came_from_segment = normalized
                normalized_lines.append(normalized_line)
                if came_from_segment:
                    split_stats.segment_lines += 1
                else:
                    split_stats.detect_lines += 1

            (target_labels / label_path.name).write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")
            split_stats.labels += 1

        stats[split] = split_stats

    return write_dataset_yaml(target_root, target_yaml), stats


def print_stats(stats: dict[str, SplitStats]) -> None:
    print("Prepared dataset:", TARGET_DATASET_ROOT)
    print("Dataset yaml:", TARGET_DATASET_YAML)
    for split in SPLITS:
        split_stats = stats[split]
        print(
            f"{split}: images={split_stats.images}, labels={split_stats.labels}, "
            f"box_lines={split_stats.detect_lines}, segment_lines_converted={split_stats.segment_lines}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the rice dataset into a pure detect dataset.")
    parser.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    args = parser.parse_args()

    _, stats = prepare_dataset()
    if not args.quiet:
        print_stats(stats)


if __name__ == "__main__":
    main()
