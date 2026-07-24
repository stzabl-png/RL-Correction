"""
AffordanceModel = backbone (Sonata) + task head -> per-point contact logit (N,1).

Keeps the backbone swappable: anything returning (B,N,C) per-point features plugs
into the same head. For MVP: SonataBackbone (frozen) + MLPHead(1232 -> 128 -> 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heads import build_head


class AffordanceModel(nn.Module):
    def __init__(self, backbone: nn.Module, head_kind: str = "mlp",
                 head_hidden: int = 128, out_channels: int = 1,
                 use_normals: bool = False):
        super().__init__()
        self.backbone = backbone
        self.use_normals = use_normals
        self.head = build_head(head_kind, in_channels=backbone.out_channels,
                               hidden=head_hidden, out_channels=out_channels)

    def forward(self, coords: torch.Tensor, normals: torch.Tensor | None = None) -> torch.Tensor:
        """coords (B,N,3), normals (B,N,3)|None -> logits (B,N) [out_channels==1 squeezed]."""
        if not self.use_normals:
            normals = None
        feat = self.backbone(coords, normals)          # (B,N,C)
        logits = self.head(feat)                        # (B,N,out)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


def build_mvp_model(cfg: dict) -> "AffordanceModel":
    """Build the MVP model from the `model` block of the yaml config."""
    m = cfg
    if m.get("backbone") == "sonata_decoder":
        from .sonata_backbone import SonataDecoderBackbone
        backbone = SonataDecoderBackbone()          # encoder frozen, decoder trainable
    else:
        from .sonata_backbone import SonataBackbone
        backbone = SonataBackbone(
            grid_size=m.get("grid_size", 0.02),
            freeze=m.get("freeze_backbone", True),
            unfreeze_last_n=m.get("unfreeze_last_n", 0),
        )
    return AffordanceModel(
        backbone, head_kind=m.get("head", "mlp"),
        head_hidden=m.get("head_hidden", 128),
        out_channels=m.get("out_channels", 1),
        use_normals=cfg.get("use_normals", False),
    )
