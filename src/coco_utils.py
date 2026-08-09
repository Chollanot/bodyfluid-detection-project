"""
coco_utils.py
Utilities for the Body-Fluid-Cell object-detection project.

Handles both datasets (CX33 microscope, Smartphone-via-eyepiece), which share the
Roboflow COCO layout:  <root>/{train,valid,test}/_annotations.coco.json  + images.

Core features
-------------
1. EDA: per-class instance counts, image counts, imbalance ratios.
2. Class dropping: remove the placeholder super-category (id 0) and any class whose
   TRAIN instance count is below `min_instances` (default 30). Rare classes with too
   few examples cannot be learned or evaluated reliably.
3. Remapping: build a contiguous 0..N-1 label map over the *kept* classes, consistent
   across train/valid/test, and consistent across both datasets' own splits.
4. COCO -> YOLO conversion (for the Ultralytics YOLO26 / YOLOv12 models).
5. A filtered-COCO writer (for RF-DETR and RTMDet, which consume COCO directly).

Everything is deterministic and re-runnable, so a crashed Colab session can just
re-run this module without corrupting state.
"""
from __future__ import annotations
import json, os, shutil, collections
from pathlib import Path

SPLITS = ("train", "valid", "test")


# --------------------------------------------------------------------------- #
# Loading / EDA
# --------------------------------------------------------------------------- #
def load_coco(root: str, split: str) -> dict:
    p = Path(root) / split / "_annotations.coco.json"
    with open(p) as f:
        return json.load(f)


def category_names(coco: dict) -> dict:
    return {c["id"]: c["name"] for c in coco["categories"]}


def class_counts(coco: dict) -> collections.Counter:
    cats = category_names(coco)
    c = collections.Counter()
    for a in coco["annotations"]:
        c[cats[a["category_id"]]] += 1
    return c


def eda_table(root: str) -> "list[dict]":
    """Return per-class rows with train/valid/test instance counts."""
    counts = {s: class_counts(load_coco(root, s)) for s in SPLITS}
    names = sorted(category_names(load_coco(root, "train")).values())
    rows = []
    for n in names:
        rows.append({
            "class": n,
            "train": counts["train"].get(n, 0),
            "valid": counts["valid"].get(n, 0),
            "test":  counts["test"].get(n, 0),
        })
    return rows


# --------------------------------------------------------------------------- #
# Class selection (imbalance management, part 1: drop ultra-rare)
# --------------------------------------------------------------------------- #
PLACEHOLDER_NAMES = {"body-fluid-cells", "bodyfluid-cells"}  # roboflow super-category


def select_classes(root: str, min_instances: int = 30) -> "list[str]":
    """Names kept after removing the placeholder and any class with
    fewer than `min_instances` TRAIN instances."""
    train_counts = class_counts(load_coco(root, "train"))
    kept = []
    for name, n in sorted(train_counts.items()):
        if name.lower() in PLACEHOLDER_NAMES:
            continue
        if n < min_instances:
            continue
        kept.append(name)
    return sorted(kept)


def dropped_classes(root: str, min_instances: int = 30) -> "list[str]":
    train_counts = class_counts(load_coco(root, "train"))
    kept = set(select_classes(root, min_instances))
    out = []
    for name in sorted(train_counts):
        if name.lower() in PLACEHOLDER_NAMES:
            continue
        if name not in kept:
            out.append((name, train_counts[name]))
    return out


def build_label_map(kept: "list[str]") -> dict:
    """name -> contiguous 0..N-1 id"""
    return {name: i for i, name in enumerate(kept)}


# --------------------------------------------------------------------------- #
# Class weights (imbalance management, part 2: weighted loss)
# --------------------------------------------------------------------------- #
def class_weights(root: str, kept: "list[str]", scheme: str = "inverse_sqrt") -> "list[float]":
    """Per-class loss weights over the KEPT classes, in label-map order.
    inverse_sqrt: w_c = sqrt(N_max / N_c), normalised to mean 1.0."""
    tc = class_counts(load_coco(root, "train"))
    counts = [max(tc.get(n, 0), 1) for n in kept]
    nmax = max(counts)
    if scheme == "inverse":
        w = [nmax / c for c in counts]
    else:  # inverse_sqrt (gentler, recommended default)
        w = [(nmax / c) ** 0.5 for c in counts]
    mean = sum(w) / len(w)
    return [round(x / mean, 4) for x in w]


# --------------------------------------------------------------------------- #
# COCO -> YOLO conversion
# --------------------------------------------------------------------------- #
def coco_to_yolo(root: str, out_dir: str, kept: "list[str]",
                 min_instances: int = 30, copy_images: bool = True) -> str:
    """Write a YOLO-format dataset containing only kept classes.
    Layout:  out_dir/images/{train,val,test}  +  out_dir/labels/{train,val,test}
    Returns path to the generated data.yaml."""
    label_map = build_label_map(kept)
    out = Path(out_dir)
    yolo_split = {"train": "train", "valid": "val", "test": "test"}
    for s in SPLITS:
        (out / "images" / yolo_split[s]).mkdir(parents=True, exist_ok=True)
        (out / "labels" / yolo_split[s]).mkdir(parents=True, exist_ok=True)

    for s in SPLITS:
        coco = load_coco(root, s)
        cats = category_names(coco)
        imgs = {im["id"]: im for im in coco["images"]}
        anns_by_img = collections.defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        for img_id, im in imgs.items():
            W, H = im["width"], im["height"]
            lines = []
            for a in anns_by_img.get(img_id, []):
                name = cats[a["category_id"]]
                if name not in label_map:
                    continue  # dropped class
                x, y, w, h = a["bbox"]  # COCO xywh (top-left)
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                nw, nh = w / W, h / H
                lines.append(f"{label_map[name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stem = Path(im["file_name"]).stem
            with open(out / "labels" / yolo_split[s] / f"{stem}.txt", "w") as f:
                f.write("\n".join(lines))
            if copy_images:
                src = Path(root) / s / im["file_name"]
                if src.exists():
                    shutil.copy(src, out / "images" / yolo_split[s] / im["file_name"])

    data_yaml = out / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write("path: " + str(out.resolve()) + "\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(kept)}\n")
        f.write("names:\n")
        for i, n in enumerate(kept):
            f.write(f"  {i}: {n}\n")
    return str(data_yaml)


# --------------------------------------------------------------------------- #
# Filtered COCO writer (for RF-DETR / RTMDet)
# --------------------------------------------------------------------------- #
def write_filtered_coco(root: str, out_dir: str, kept: "list[str]",
                        copy_images: bool = True) -> str:
    """Rewrite COCO json keeping only kept classes, re-indexed to 1..N
    (COCO category ids conventionally start at 1). Copies images.
    Returns out_dir."""
    label_map = {name: i + 1 for i, name in enumerate(kept)}  # 1-based for COCO
    out = Path(out_dir)
    for s in SPLITS:
        (out / s).mkdir(parents=True, exist_ok=True)
        coco = load_coco(root, s)
        cats = category_names(coco)
        new = {
            "images": coco["images"],
            "categories": [{"id": label_map[n], "name": n, "supercategory": "cell"}
                           for n in kept],
            "annotations": [],
        }
        nid = 1
        for a in coco["annotations"]:
            name = cats[a["category_id"]]
            if name not in label_map:
                continue
            b = dict(a)
            b["category_id"] = label_map[name]
            b["id"] = nid; nid += 1
            new["annotations"].append(b)
        with open(out / s / "_annotations.coco.json", "w") as f:
            json.dump(new, f)
        if copy_images:
            for im in coco["images"]:
                src = Path(root) / s / im["file_name"]
                if src.exists():
                    shutil.copy(src, out / s / im["file_name"])
    return str(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--min_instances", type=int, default=30)
    a = ap.parse_args()
    kept = select_classes(a.root, a.min_instances)
    print(f"Kept {len(kept)} classes:", kept)
    print("Dropped:", dropped_classes(a.root, a.min_instances))
    print("Weights:", class_weights(a.root, kept))
