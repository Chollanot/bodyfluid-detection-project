"""
novel_loss.py
Class-imbalance-aware loss for the NOVEL detector (BFCell-Attn), Novelty (2).

Ultralytics' default detection loss uses a plain BCE for the classification term,
which lets abundant classes (Neutrophil, Macrophage, Lymphocyte: thousands of
instances) dominate the gradient and starves rare classes (bacteria, signet-ring,
mitotic: tens of instances). We replace that BCE with a **weighted focal BCE**:

  loss = w_c * (1 - p_t)^gamma * BCE(pred, target)

  * w_c    : per-class weight from coco_utils.class_weights (inverse-sqrt frequency),
             up-weights rare classes.
  * focal  : (1 - p_t)^gamma focuses training on hard / under-learned examples.

apply_imbalance_loss() patches a model instance in place, so the rest of the
Ultralytics training loop (checkpointing, resume, validation) is unchanged.
"""
from __future__ import annotations
import types
import torch
import torch.nn as nn


class WeightedFocalBCE(nn.Module):
    """Elementwise (reduction='none') weighted focal BCE, drop-in for the
    `self.bce` used inside v8DetectionLoss. Returns a tensor the same shape as
    the input so the caller's own reduction/normalisation still applies."""
    def __init__(self, class_weights, gamma: float = 1.5):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(class_weights, dtype=torch.float32))
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target):
        # pred/target: (batch, anchors, num_classes)
        ce = self.bce(pred, target)
        p = torch.sigmoid(pred)
        p_t = target * p + (1 - target) * (1 - p)
        focal = (1.0 - p_t).clamp(min=1e-6) ** self.gamma
        w = self.w.to(pred.device)
        while w.dim() < pred.dim():
            w = w.unsqueeze(0)
        return ce * focal * w


def apply_imbalance_loss(model, class_weights, gamma: float = 1.5):
    """Patch an Ultralytics YOLO model so its detection loss uses the weighted
    focal BCE. Call AFTER constructing the model, BEFORE model.train(...)."""
    from ultralytics.utils.loss import v8DetectionLoss

    cw = list(class_weights)

    def init_criterion(det_model):
        crit = v8DetectionLoss(det_model)
        crit.bce = WeightedFocalBCE(cw, gamma)
        return crit

    det_model = model.model  # DetectionModel (nn.Module)
    det_model.init_criterion = types.MethodType(init_criterion, det_model)
    # force rebuild on next forward
    if hasattr(det_model, "criterion"):
        det_model.criterion = None
    print(f"[novel_loss] weighted focal BCE attached (gamma={gamma}, "
          f"{len(cw)} class weights).")
    return model
