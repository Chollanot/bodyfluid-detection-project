"""
predict_export.py
Turn each model's raw outputs into the ONE common evaluation format used by
metrics.py, and read YOLO-format ground truth. Keeping this in one place means
all six models are scored identically and fairly.

common record formats
  gt   : {"image_id": stem, "class_id": int(0..K-1), "bbox":[x1,y1,x2,y2]}
  pred : {"image_id": stem, "class_id": int, "score": float, "bbox":[x1,y1,x2,y2]}
"""
from __future__ import annotations
import os, glob, json
from pathlib import Path


def read_yolo_gt(images_dir: str, labels_dir: str):
    import cv2
    gts = []
    for lp in glob.glob(os.path.join(labels_dir, "*.txt")):
        stem = Path(lp).stem
        ips = glob.glob(os.path.join(images_dir, stem + ".*"))
        if not ips:
            continue
        h, w = cv2.imread(ips[0]).shape[:2]
        for line in open(lp):
            if not line.strip():
                continue
            c, cx, cy, bw, bh = map(float, line.split())
            gts.append({"image_id": stem, "class_id": int(c),
                        "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                 (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return gts


def read_coco_gt(coco_json: str, cat_id_to_contiguous: dict):
    """cat_id_to_contiguous maps COCO category_id -> 0..K-1 label id."""
    d = json.load(open(coco_json))
    id2name = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    gts = []
    for a in d["annotations"]:
        if a["category_id"] not in cat_id_to_contiguous:
            continue
        x, y, w, h = a["bbox"]
        gts.append({"image_id": id2name[a["image_id"]],
                    "class_id": cat_id_to_contiguous[a["category_id"]],
                    "bbox": [x, y, x + w, y + h]})
    return gts


def ultralytics_predictions(model, images_dir, imgsz=640, conf=0.001):
    """model: an Ultralytics YOLO. Returns common-format predictions."""
    preds = []
    for ip in glob.glob(os.path.join(images_dir, "*.*")):
        stem = Path(ip).stem
        r = model.predict(ip, imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is None:
            continue
        for b, s, c in zip(r.boxes.xyxy.cpu().numpy(),
                           r.boxes.conf.cpu().numpy(),
                           r.boxes.cls.cpu().numpy()):
            preds.append({"image_id": stem, "class_id": int(c), "score": float(s),
                          "bbox": [float(x) for x in b]})
    return preds


def save_bundle(out_json, gts, preds, classes, train_time_sec=None):
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ground_truth": gts, "predictions": preds, "classes": classes,
               "train_time_sec": train_time_sec}, open(out_json, "w"))
    print(f"[predict_export] {len(preds)} preds / {len(gts)} gt -> {out_json}")
    return out_json
