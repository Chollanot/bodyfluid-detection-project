"""
xai_utils.py
Explainable-AI (XAI) for the detectors.

Primary method: EigenCAM (works without class gradients, robust for detection
backbones) plus Grad-CAM when a target layer exposes gradients. Built on the
`grad-cam` package (pip install grad-cam). Produces a heatmap overlaid on the
input image that shows which regions drove the model's detections.

The helper is written for Ultralytics models (novel model, YOLO26, YOLOv12) whose
`.model` is a torch nn.Module. For RF-DETR / EfficientDet / RTMDet the same EigenCAM
idea applies to their backbone; a `generic_eigencam` entry point is provided.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def _to_rgb_float(img_bgr):
    import cv2
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.float32(rgb) / 255.0


def ultralytics_eigencam(weights, image_path, out_path,
                         target_layer_index=-2, imgsz=640):
    """EigenCAM on an Ultralytics model (YOLO26/YOLOv12/novel).
    target_layer_index indexes model.model.model (backbone/neck)."""
    import cv2, torch
    from ultralytics import YOLO
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    yolo = YOLO(weights)
    torch_model = yolo.model.eval()
    device = next(torch_model.parameters()).device
    target_layers = [list(torch_model.model)[target_layer_index]]

    img = cv2.resize(cv2.imread(str(image_path)), (imgsz, imgsz))
    rgb = _to_rgb_float(img)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    cam = EigenCAM(model=torch_model, target_layers=target_layers)
    # A YOLO detection model returns a tuple, so BaseCAM's default argmax on the
    # output crashes ('tuple' has no .cpu()). EigenCAM ignores the target (it works
    # from activations), so we pass a dummy target to skip that code path.
    grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]
    vis = show_cam_on_image(rgb, grayscale, use_rgb=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return str(out_path)


def generic_eigencam(torch_model, target_layers, image_path, out_path, imgsz=640):
    """Framework-agnostic EigenCAM given a torch model and target layer list."""
    import cv2, torch
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    img = cv2.resize(cv2.imread(str(image_path)), (imgsz, imgsz))
    rgb = _to_rgb_float(img)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    cam = EigenCAM(model=torch_model, target_layers=target_layers)
    grayscale = cam(input_tensor=tensor)[0]
    vis = show_cam_on_image(rgb, grayscale, use_rgb=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return str(out_path)
