"""
reporting.py
Unified per-model reporting for the six-model benchmark.

Adds the detection-native and efficiency metrics that classification-style
metrics (in metrics.py) don't cover, computed IDENTICALLY for every model from
the same common prediction bundles:

  * mAP@0.5 and mAP@0.5:0.95   (COCO-style, from the saved gt/pred bundles)
  * per-class AP@0.5
  * parameters (M) and inference FPS  (read from each model's *_meta.json,
    which its notebook writes at export time — helpers below)

It also builds the full comparison table (classification metrics from
metrics.summarize + mAP + params + FPS + training time) and renders a one-page
"model card" per model (loss curves + per-class F1 + confusion matrix).

Everything is derived from files already on Drive, so it re-runs after any crash.
"""
from __future__ import annotations
import os, json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

import metrics as M   # same src/ folder


# --------------------------------------------------------------------------- #
# mAP (COCO-style) from the common bundle format
# --------------------------------------------------------------------------- #
def _ap_per_class(gt, preds, cls, iou_thr):
    """VOC/COCO greedy-match AP for one class at one IoU threshold.
    Returns None if the class has no ground-truth instances (excluded from mAP)."""
    gt_by_img = defaultdict(list)
    npos = 0
    for g in gt:
        if g["class_id"] == cls:
            gt_by_img[g["image_id"]].append(g["bbox"]); npos += 1
    if npos == 0:
        return None
    dets = sorted((p for p in preds if p["class_id"] == cls),
                  key=lambda x: -x["score"])
    if not dets:
        return 0.0
    matched = defaultdict(set)
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for i, d in enumerate(dets):
        gts = gt_by_img.get(d["image_id"], [])
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gts):
            if j in matched[d["image_id"]]:
                continue
            iou = M.iou_xyxy(d["bbox"], gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = 1; matched[d["image_id"]].add(best_j)
        else:
            fp[i] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / npos
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    # 101-point interpolation (COCO)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 101.0
    return float(ap)


def map_at_iou(gt, preds, num_classes, iou_thr=0.5):
    aps = [_ap_per_class(gt, preds, c, iou_thr) for c in range(num_classes)]
    aps = [a for a in aps if a is not None]
    return float(np.mean(aps)) if aps else float("nan")


def per_class_ap50(gt, preds, num_classes, class_names):
    out = {}
    for c in range(num_classes):
        a = _ap_per_class(gt, preds, c, 0.5)
        out[class_names[c]] = (float("nan") if a is None else a)
    return out


def map_50_95(gt, preds, num_classes):
    thrs = np.round(np.arange(0.5, 1.0, 0.05), 2)
    vals = [map_at_iou(gt, preds, num_classes, t) for t in thrs]
    vals = [v for v in vals if v == v]  # drop nan
    return float(np.mean(vals)) if vals else float("nan")


def detection_scores(bundle):
    """bundle = loaded predictions/<model>.json (has ground_truth, predictions, classes)."""
    gt, preds, classes = bundle["ground_truth"], bundle["predictions"], bundle["classes"]
    K = len(classes)
    return {
        "mAP@0.5": map_at_iou(gt, preds, K, 0.5),
        "mAP@0.5:0.95": map_50_95(gt, preds, K),
        "AP50_per_class": per_class_ap50(gt, preds, K, classes),
    }


# --------------------------------------------------------------------------- #
# Efficiency helpers (call these inside each model's notebook at export time)
# --------------------------------------------------------------------------- #
def count_params_millions(torch_module) -> float:
    return round(sum(p.numel() for p in torch_module.parameters()) / 1e6, 2)


def measure_fps(predict_fn, sample_images, warmup=3, runs=20) -> float:
    """predict_fn(path)->anything. Times inference over sample images (single image)."""
    import time
    imgs = (sample_images * ((warmup + runs) // max(len(sample_images), 1) + 1))
    for p in imgs[:warmup]:
        predict_fn(p)
    t0 = time.time(); n = 0
    for p in imgs[warmup:warmup + runs]:
        predict_fn(p); n += 1
    dt = time.time() - t0
    return round(n / dt, 2) if dt > 0 else float("nan")


def save_meta(root, model_key, params_M=None, fps=None):
    p = os.path.join(root, "metrics", f"{model_key}_meta.json")
    json.dump({"params_M": params_M, "fps": fps}, open(p, "w"))
    return p


def _load_meta(root, model_key):
    p = os.path.join(root, "metrics", f"{model_key}_meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _load_time(root, model_key, bundle):
    if bundle.get("train_time_sec") is not None:
        return bundle["train_time_sec"]
    p = os.path.join(root, "metrics", f"{model_key}_time.json")
    return json.load(open(p))["train_time_sec"] if os.path.exists(p) else float("nan")


# --------------------------------------------------------------------------- #
# Full comparison table (classification + detection + efficiency)
# --------------------------------------------------------------------------- #
def build_full_comparison(root, model_map, dataset, save_csv=True, score_thr=0.25):
    """model_map: {display_name: file_key}. Reads predictions/<key>.json for each,
    returns a pandas DataFrame with every headline metric, sorted by mAP@0.5:0.95.

    Classification metrics (Sensitivity/Specificity/F1/MCC) are computed at a
    confidence OPERATING POINT `score_thr` (default 0.25 — the standard for a
    detector confusion matrix), so low-confidence junk boxes don't flood the
    false-positive count. mAP and AUC use the full score range."""
    import pandas as pd, math
    rows = []
    for disp, key in model_map.items():
        bp = os.path.join(root, "predictions", f"{key}.json")
        if not os.path.exists(bp):
            print("MISSING (skip):", disp); continue
        bundle = json.load(open(bp))
        gt, preds, classes = bundle["ground_truth"], bundle["predictions"], bundle["classes"]
        K = len(classes)
        # classification metrics at the operating point (fewer junk false positives)
        cm_op, _ = M.evaluate(gt, preds, K, iou_thr=0.5, score_thr=score_thr)
        pcm = M.per_class_metrics(cm_op, classes)
        macro = {k: float(np.mean([pcm[n][k] for n in classes]))
                 for k in ("Sensitivity", "Specificity", "F1", "MCC")}
        # AUC from the FULL score range
        _, roc_full = M.evaluate(gt, preds, K, iou_thr=0.5, score_thr=0.001)
        aucs = M.roc_auc_per_class(roc_full, K, classes)
        valid = [aucs[n]["auc"] for n in classes if not math.isnan(aucs[n]["auc"])]
        auc = float(np.mean(valid)) if valid else float("nan")
        det = detection_scores(bundle)
        meta = _load_meta(root, key)
        rows.append({
            "Model": disp,
            "mAP@0.5": det["mAP@0.5"],
            "mAP@0.5:0.95": det["mAP@0.5:0.95"],
            "Sensitivity": macro["Sensitivity"],
            "Specificity": macro["Specificity"],
            "F1": macro["F1"],
            "MCC": macro["MCC"],
            "AUC": auc,
            "Params (M)": meta.get("params_M"),
            "FPS": meta.get("fps"),
            "Train time (min)": round(_load_time(root, key, bundle) / 60, 2),
        })
    df = pd.DataFrame(rows).sort_values("mAP@0.5:0.95", ascending=False).reset_index(drop=True)
    for c in df.columns[1:]:
        df[c] = df[c].apply(lambda v: round(v, 4) if isinstance(v, float) else v)
    if save_csv:
        out = os.path.join(root, "metrics", "full_comparison_table.csv")
        df.to_csv(out, index=False); print("saved", out)
    return df


# --------------------------------------------------------------------------- #
# Per-model "model card" figure
# --------------------------------------------------------------------------- #
def _read_loss_history(root, key):
    """Return (epochs, {loss_name: values}) from whichever log the framework wrote."""
    # Ultralytics: results.csv
    rc = os.path.join(root, "checkpoints", key, "results.csv")
    if os.path.exists(rc):
        import pandas as pd
        d = pd.read_csv(rc); d.columns = [c.strip() for c in d.columns]
        losses = {c.split("/")[-1]: d[c].tolist()
                  for c in d.columns if "loss" in c and c.startswith("train")}
        return d["epoch"].tolist(), losses
    # EfficientDet / Faster R-CNN: history.json
    hj = os.path.join(root, "checkpoints", key, "history.json")
    if os.path.exists(hj):
        h = json.load(open(hj))
        ep = [r["epoch"] for r in h]
        keys = [k for k in h[0].keys() if k not in ("epoch",)]
        return ep, {k: [r.get(k) for r in h] for k in keys}
    return None, {}


def render_model_card(root, disp, key, out_path=None):
    """One-page PNG: loss curves + per-class F1 bar + confusion matrix."""
    import matplotlib.pyplot as plt
    bundle = json.load(open(os.path.join(root, "predictions", f"{key}.json")))
    classes = bundle["classes"]; K = len(classes)
    cm, roc = M.evaluate(bundle["ground_truth"], bundle["predictions"], K, iou_thr=0.5, score_thr=0.25)
    summ = M.summarize(cm, roc, classes)
    f1 = {n: summ["per_class"][n]["F1"] for n in classes}

    fig = plt.figure(figsize=(15, 4.6))
    # 1) loss curves
    ax1 = fig.add_subplot(1, 3, 1)
    ep, losses = _read_loss_history(root, key)
    if ep and losses:
        for name, vals in losses.items():
            ax1.plot(ep, vals, label=name)
        ax1.set_title(f"{disp}: training loss"); ax1.set_xlabel("epoch"); ax1.legend(fontsize=7)
    else:
        ax1.text(0.5, 0.5, "loss log not found", ha="center"); ax1.set_axis_off()
    # 2) per-class F1
    ax2 = fig.add_subplot(1, 3, 2)
    names = sorted(f1, key=lambda n: -f1[n])
    ax2.barh(names, [f1[n] for n in names]); ax2.invert_yaxis()
    ax2.set_title("per-class F1"); ax2.set_xlim(0, 1); ax2.tick_params(labelsize=6)
    # 3) confusion matrix (normalised, cropped to classes)
    ax3 = fig.add_subplot(1, 3, 3)
    cmn = cm[:K, :K].astype(float)
    cmn = cmn / np.maximum(cmn.sum(axis=0, keepdims=True), 1)
    im = ax3.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax3.set_title("confusion (col-normalised)"); ax3.set_xlabel("true"); ax3.set_ylabel("pred")
    fig.colorbar(im, ax=ax3, fraction=0.046)
    fig.suptitle(f"Model card — {disp}   "
                 f"(mAP50={detection_scores(bundle)['mAP@0.5']:.3f}, "
                 f"MCC={summ['macro']['MCC']:.3f})")
    fig.tight_layout()
    out_path = out_path or os.path.join(root, "figures", f"model_card_{key}.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print("saved", out_path)
    return out_path


def render_confusion_matrix(root, disp, key, normalize="true", out_path=None, annotate=True, score_thr=0.25):
    """Full-size, labelled, per-model confusion matrix (readable).
    normalize: "true"  -> column-normalised (diagonal = recall/sensitivity per true class)
               "pred"  -> row-normalised    (diagonal = precision per predicted class)
               None    -> raw counts
    Includes a 'background' row/col (missed GT = false negatives; false positives)."""
    import matplotlib.pyplot as plt
    bundle = json.load(open(os.path.join(root, "predictions", f"{key}.json")))
    classes = bundle["classes"]; Kc = len(classes)
    cm, _ = M.evaluate(bundle["ground_truth"], bundle["predictions"], Kc, iou_thr=0.5, score_thr=score_thr)
    labels = classes + ["background"]          # cm: rows=pred, cols=true, last index=background
    mat = cm.astype(float)
    if normalize == "true":
        d = mat.sum(axis=0, keepdims=True); d[d == 0] = 1; shown = mat / d
        suffix = "(column-normalised — diagonal = recall per true class)"
    elif normalize == "pred":
        d = mat.sum(axis=1, keepdims=True); d[d == 0] = 1; shown = mat / d
        suffix = "(row-normalised — diagonal = precision per predicted class)"
    else:
        shown = mat; suffix = "(raw counts)"
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * n), max(8, 0.55 * n)))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=(1 if normalize else None))
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("True class"); ax.set_ylabel("Predicted class")
    ax.set_title(f"{disp} — confusion matrix  {suffix}")
    if annotate and n <= 26:
        for i in range(n):
            for j in range(n):
                v = shown[i, j]
                if v > 0.005:
                    txt = f"{v:.2f}" if normalize else f"{int(mat[i, j])}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                            color="white" if v > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path = out_path or os.path.join(root, "figures", f"confusion_{key}.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160); plt.close(fig)
    print("saved", out_path)
    return out_path
