"""
novel_modules.py
Custom attention module for the NOVEL detector (BFCell-Attn).

CBAMLite = Convolutional Block Attention Module, implemented to be **shape- and
scale-invariant**: it infers the channel count from the input tensor at runtime
(lazy layers), so it drops into any Ultralytics YOLO scale (n/s/m/l/x) without the
YAML channel-scaling pitfalls that affect hand-written custom graphs. Output shape
== input shape (attention is multiplicative), so it never disturbs downstream layers.

It combines:
  * Channel attention  (SE-style squeeze-and-excite over avg+max pooled descriptors)
  * Spatial attention  (7x7 conv over channel-pooled maps)

register_bfcell_modules() injects CBAMLite into the namespaces Ultralytics' YAML
parser looks in, so `- [16, 1, CBAMLite, []]` resolves. Call it once before
YOLO('bfcell_attn.yaml').
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, reduction: int = 16):
        super().__init__()
        self.reduction = reduction
        self.mlp = None  # built lazily
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.act = nn.Sigmoid()

    def _build(self, c: int, device):
        hidden = max(c // self.reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
        ).to(device)

    def forward(self, x):
        if self.mlp is None:
            self._build(x.shape[1], x.device)
        out = self.mlp(self.avg(x)) + self.mlp(self.max(x))
        return x * self.act(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = self.act(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAMLite(nn.Module):
    """Scale-invariant CBAM. Accepts arbitrary positional/keyword args so it is
    robust to however the YOLO YAML parser forwards channel arguments."""
    def __init__(self, *args, reduction: int = 16, kernel_size: int = 7, **kwargs):
        super().__init__()
        # tolerate parser passing kernel size as 2nd positional arg
        if len(args) >= 2 and isinstance(args[1], int):
            kernel_size = args[1]
        self.channel = ChannelAttention(reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))


def register_bfcell_modules():
    """Make CBAMLite discoverable by the Ultralytics YAML parser.

    parse_model resolves a layer's module by looking up its name in the global
    namespace of ultralytics.nn.tasks, so adding CBAMLite to that module's
    __dict__ is sufficient. We also add it to ultralytics.nn.modules."""
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    tasks.CBAMLite = CBAMLite
    modules.CBAMLite = CBAMLite
    print("[novel_modules] CBAMLite registered for YAML parsing.")
