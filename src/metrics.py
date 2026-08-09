"""
metrics.py
Unified evaluation for object detection expressed as classification-style metrics.

WHY THIS EXISTS
---------------
Sensitivity, Specificity, F1, MCC, ROC and AUC are classification metrics, but the
task is detection. To report them fairly and identically across all six models, we:

1. Match predicted boxes to ground-truth boxes greedily by descending confidence at
   a fixed IoU threshold (default 0.5), per class, per image.
2. Accumulate a (K+1) x (K+1) confusion matrix over the K kept classes plus a
   "background" row/column (unmatched GT -> background column = false negative;
   unmatched prediction -> background row = false positive). This mirrors the
   confusion matrix Ultralytics/COCO tooling produces.
3. Derive per-class one-vs-rest TP/FP/FN/TN from that matrix, then compute
   Sensitivity(=Recall), Specificity, Precision, F1 and MCC.
4. For ROC/AUC, rank each class's predictions by confidence and label them TP(1) /
   FP(0) by IoU match; the ROC curve then measures how well confidence separates
   correct from incorrect detections (a standard detector-level ROC).

Every model in the project produces predictions in ONE common format, so this module
is the single source of truth for the comparison. All raw arrays are saved to disk so
a Colab crash never loses computed results.

Common formats
--------------
ground_truth : list[ {"image_id": str|int, "class_id": int, "bbox": [x1,y1,x2,y2]} ]
predictions  : list[ {"image_id": str|int, "class_id": int, "score": float,
                       "bbox": [x1,y1,x2,y2]} ]
class ids are the contiguous 0..K-1 label-map ids from coco_utils.
"""
from __future__ import annotations
import json, math, collections
from pathlib import Path
import numpy as np


# --------------------------------------------------------------------------- #
def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


# --------------------------------------------------------------------------- #
def evaluate(ground_truth, predictions, num_classes,
             iou_thr=0.5, score_thr=0.001):
    """Return confusion matrix (K+1 sq) and per-detection (score,is_tp,class) records."""
    K = num_classes
    cm = np.zeros((K + 1, K + 1), dtype=np.int64)  # rows=pred, cols=gt; index K = background

    gt_by_img = collections.defaultdict(list)
    for g in ground_truth:
        gt_by_img[g["image_id"]].append(g)
    pr_by_img = collections.defaultdict(list)
    for p in predictions:
        if p["score"] >= score_thr:
            pr_by_img[p["image_id"]].append(p)

    roc_records = []  # (class_id, score, is_tp)

    all_imgs = set(gt_by_img) | set(pr_by_img)
    for img in all_imgs:
        gts = gt_by_img.get(img, [])
        prs = sorted(pr_by_img.get(img, []), key=lambda x: -x["score"])
        matched_gt = set()
        for p in prs:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if j in matched_gt:
                    continue
                i = iou_xyxy(p["bbox"], g["bbox"])
                if i >= iou_thr and i > best_iou:
                    best_iou, best_j = i, j
            if best_j >= 0:
                g = gts[best_j]
                matched_gt.add(best_j)
                cm[p["class_id"], g["class_id"]] += 1
                roc_records.append((p["class_id"], p["score"],
                                    1 if p["class_id"] == g["class_id"] else 0))
            else:
                cm[p["class_id"], K] += 1  # false positive -> background col
                roc_records.append((p["class_id"], p["score"], 0))
        for j, g in enumerate(gts):
            if j not in matched_gt:
                cm[K, g["class_id"]] += 1  # false negative -> background row
    return cm, roc_records


# --------------------------------------------------------------------------- #
def per_class_metrics(cm, class_names):
    """One-vs-rest metrics from the confusion matrix (excludes background index)."""
    K = len(class_names)
    total = cm.sum()
    out = {}
    for c in range(K):
        tp = cm[c, c]
        fp = cm[c, :].sum() - tp          # predicted c, truth not c (incl background)
        fn = cm[:, c].sum() - tp          # truth c, predicted not c (incl background)
        tn = total - tp - fp - fn
        sens = tp / (tp + fn) if (tp + fn) else 0.0          # recall / sensitivity
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
        out[class_names[c]] = dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                                   Sensitivity=sens, Specificity=spec, Precision=prec,
                                   F1=f1, MCC=mcc)
    return out


def roc_auc_per_class(roc_records, num_classes, class_names):
    """Detector-level ROC/AUC per class from (class,score,is_tp) records.
    Returns {class: {'auc':.., 'fpr':[...], 'tpr':[...]}}."""
    by_c = collections.defaultdict(list)
    for c, s, y in roc_records:
        by_c[c].append((s, y))
    res = {}
    for c in range(num_classes):
        recs = sorted(by_c.get(c, []), key=lambda x: -x[0])
        P = sum(y for _, y in recs)
        N = len(recs) - P
        if P == 0 or N == 0:
            res[class_names[c]] = dict(auc=float("nan"), fpr=[0, 1], tpr=[0, 1])
            continue
        tp = fp = 0
        fpr, tpr = [0.0], [0.0]
        auc = 0.0
        prev_fpr = 0.0
        for s, y in recs:
            if y == 1:
                tp += 1
            else:
                fp += 1
            cur_tpr, cur_fpr = tp / P, fp / N
            auc += (cur_fpr - prev_fpr) * cur_tpr  # trapezoid-ish (right Riemann)
            prev_fpr = cur_fpr
            tpr.append(cur_tpr); fpr.append(cur_fpr)
        res[class_names[c]] = dict(auc=auc, fpr=fpr, tpr=tpr)
    return res


# --------------------------------------------------------------------------- #
def summarize(cm, roc, class_names):
    pcm = per_class_metrics(cm, class_names)
    aucs = roc_auc_per_class(roc, len(class_names), class_names)
    def macro(key):
        vals = [pcm[n][key] for n in class_names]
        return float(np.mean(vals))
    valid_auc = [aucs[n]["auc"] for n in class_names if not math.isnan(aucs[n]["auc"])]
    return {
        "per_class": pcm,
        "roc": aucs,
        "macro": {
            "Sensitivity": macro("Sensitivity"),
            "Specificity": macro("Specificity"),
            "Precision": macro("Precision"),
            "F1": macro("F1"),
            "MCC": macro("MCC"),
            "AUC": float(np.mean(valid_auc)) if valid_auc else float("nan"),
        },
    }


def save_results(model_name, dataset, summary, train_time_sec, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, "dataset": dataset,
               "train_time_sec": train_time_sec, **summary}
    with open(out / f"{model_name}_{dataset}_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    return str(out / f"{model_name}_{dataset}_metrics.json")


if __name__ == "__main__":
    # tiny self-test
    gt = [{"image_id": 1, "class_id": 0, "bbox": [0, 0, 10, 10]},
          {"image_id": 1, "class_id": 1, "bbox": [20, 20, 30, 30]}]
    pr = [{"image_id": 1, "class_id": 0, "score": 0.9, "bbox": [0, 0, 9, 9]},
          {"image_id": 1, "class_id": 1, "score": 0.3, "bbox": [21, 21, 29, 29]},
          {"image_id": 1, "class_id": 0, "score": 0.4, "bbox": [50, 50, 60, 60]}]
    cm, roc = evaluate(gt, pr, 2)
    s = summarize(cm, roc, ["A", "B"])
    print("macro:", s["macro"])
    print("A:", {k: round(v, 3) if isinstance(v, float) else v
                 for k, v in s["per_class"]["A"].items()})
