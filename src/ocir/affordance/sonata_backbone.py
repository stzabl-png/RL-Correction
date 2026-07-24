"""
Sonata (encoder-only pretrained PTv3) backbone wrapper.  VERIFIED WORKING on
RTX 5090 / sm_120 with spconv-cu126 + torch_scatter, enable_flash=False.

Verified facts (from a real forward on the 5090):
  * pretrained in_channels = 9  == coord(3) + color(3) + normal(3)  [FIXED]
  * after walking the pooling chain, per-ORIGINAL-point feature dim = 1232
  * load with custom_config(enable_flash=False, enc_patch_size=[1024]*5)

Input-channel convention for OUR task (no color in any of our datasets):
  * "xyz-only"     -> feat = [coord, gray(3 color), zeros(3 normal)]
  * "xyz+normals"  -> feat = [coord, gray(3 color), normal]
  `use_normals` just toggles the normal block.

COLOR POLICY (updated after clarification): contact affordance is GEOMETRY-driven,
not appearance-driven (where you grasp a mug does not depend on its colour), and
our training set (AffordanceNet) has NO colour to learn from. So we train the
model to be COLOUR-INVARIANT: feed a constant neutral mid-gray (127.5 -> 0.5 after
Sonata's /255) on the colour channel. Deployment target = Meta SAM 3D, which
reconstructs a TEXTURED mesh, so real RGB WOULD be available — but we must feed
the SAME gray at deploy that we trained on, otherwise a gray-train / colour-deploy
mismatch hurts. i.e. we deliberately DISCARD SAM 3D's texture on the colour channel
and rely on geometry(+normals). (If we ever want to exploit texture, we'd have to
add colour augmentation in training first — but AffordanceNet gives no colour
signal to learn from, so it would not help on this dataset.)

SCALE CAVEAT (a real hyperparameter to sweep in T6): Sonata was pretrained on
metre-scale indoor scans with GridSample(0.02). Our objects are ~0.1-0.3 m, so
feeding raw metres at grid 0.02 gives only a handful of voxels. We therefore feed
UNIT-SPHERE-NORMALIZED coords (done in the dataloader) and keep grid ~0.02, which
yields ~100 voxels across the object. `grid_size` and the input scale are knobs.

Batching: PTv3 uses an offset/batch layout, not a (B,N,3) tensor. For MVP we loop
over the batch (frozen forward is cheap) and stack. Optimize with real collation
later if throughput matters.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import sonata


class SonataBackbone(nn.Module):
    OUT_CHANNELS = 1232   # verified per-original-point feature dim after upsample

    def __init__(self, ckpt: str = "sonata", repo_id: str = "facebook/sonata",
                 grid_size: float = 0.02, freeze: bool = True,
                 unfreeze_last_n: int = 0, enc_patch_size: int = 1024,
                 color_value: float = 127.5):
        super().__init__()
        # Sonata's NormalizeColor does color/255 and was pretrained on real RGB
        # (ScanNet, mean ~120). Our datasets have no color, so we fill a NEUTRAL
        # MID-GRAY (127.5 -> 0.5 after /255), which sits in-distribution instead
        # of black (0). At deployment on SAM3D-reconstructed clouds, pass real RGB.
        self.color_value = color_value
        self.grid_size = grid_size
        self.freeze = freeze
        self.unfreeze_last_n = unfreeze_last_n
        self.out_channels = self.OUT_CHANNELS

        self.model = sonata.load(
            ckpt, repo_id=repo_id,
            custom_config=dict(enc_patch_size=[enc_patch_size] * 5, enable_flash=False),
        )
        self._transform = sonata.transform.default()   # CenterShift+GridSample(0.02)+...
        self._set_trainable()

    def _set_trainable(self):
        """
        Freeze everything, then (for partial fine-tune) unfreeze the DEEPEST N
        encoder stages. PTv3 encoder = self.model.enc with stages enc0..enc4;
        enc4 is deepest/most task-relevant, enc0 is the shallow general-geometry
        stem. We unfreeze from the deep end: N=1 -> enc4, N=2 -> enc3+enc4, ...
        The embedding stem stays frozen. Layer-wise LR is applied in train.py
        (head lr vs backbone lr) over whatever parameters end up requires_grad.
        """
        for p in self.model.parameters():
            p.requires_grad = False
        self.unfrozen_modules = []
        if self.unfreeze_last_n > 0:
            enc_stages = list(self.model.enc.children())          # [enc0..enc4]
            for stage in enc_stages[-self.unfreeze_last_n:]:      # deepest N
                for p in stage.parameters():
                    p.requires_grad = True
                self.unfrozen_modules.append(stage)
        self.fully_frozen = len(self.unfrozen_modules) == 0
        n_tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[SonataBackbone] unfreeze_last_n={self.unfreeze_last_n} -> "
              f"{n_tr/1e6:.2f}M backbone params trainable, fully_frozen={self.fully_frozen}")
        self.model.eval()   # start in eval; train() re-enables the unfrozen stages

    def train(self, mode: bool = True):
        """Keep FROZEN parts in eval() (fixed norm stats) even when the parent
        AffordanceModel is in train mode; only the unfrozen deep stages train."""
        super().train(mode)
        self.model.eval()                       # all frozen stats fixed by default
        if mode and not self.fully_frozen:
            for stage in self.unfrozen_modules:  # re-enable just the unfrozen stages
                stage.train(mode)
        return self

    def _transform_one(self, coord: np.ndarray, normal: np.ndarray | None) -> dict:
        N = coord.shape[0]
        color = np.full((N, 3), self.color_value, np.float32)   # neutral mid-gray
        normal = np.zeros((N, 3), np.float32) if normal is None else normal.astype(np.float32)
        return self._transform({"coord": coord.copy().astype(np.float32),
                                "color": color, "normal": normal})

    def _run_model(self, data_dict: dict) -> torch.Tensor:
        """Forward + walk pooling chain. Returns grid-level feat (M_total, 1232)."""
        out = self.model(data_dict)
        while "pooling_parent" in out.keys():
            parent = out.pop("pooling_parent")
            inv = out.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, out.feat[inv]], dim=-1)
            out = parent
        return out.feat

    def forward(self, coords: torch.Tensor, normals: torch.Tensor | None = None) -> torch.Tensor:
        """
        coords:  (B,N,3) unit-sphere-normalized xyz
        normals: (B,N,3) or None
        returns: (B,N,1232) per-original-point features

        BATCHED: transform each cloud (CPU), then concatenate into ONE point dict
        with a cumulative `offset` so PTv3 processes the whole batch in a single
        forward (PTv3 disambiguates clouds via the batch bits in serialization).
        ~5-10x faster than a per-sample loop.
        """
        B, N, _ = coords.shape
        dev = next(self.model.parameters()).device
        coord_l, grid_l, color_l, feat_l, inv_l = [], [], [], [], []
        counts, cum = [], 0
        for b in range(B):
            c = coords[b].detach().cpu().numpy().astype(np.float32)
            nrm = None if normals is None else normals[b].detach().cpu().numpy()
            p = self._transform_one(c, nrm)
            m = p["coord"].shape[0]
            coord_l.append(p["coord"]); grid_l.append(p["grid_coord"])
            color_l.append(p["color"]); feat_l.append(p["feat"])
            inv_l.append(p["inverse"] + cum)   # index into the CONCATENATED grid feat
            counts.append(m); cum += m
        offset = torch.tensor(np.cumsum(counts), dtype=torch.long)
        data = {
            "coord": torch.cat(coord_l).to(dev),
            "grid_coord": torch.cat(grid_l).to(dev),
            "color": torch.cat(color_l).to(dev),
            "feat": torch.cat(feat_l).to(dev),
            "offset": offset.to(dev),
        }
        global_inv = torch.cat(inv_l).to(dev)   # (B*N,) into concatenated grid
        ctx = torch.no_grad() if self.fully_frozen else torch.enable_grad()
        with ctx:
            grid_feat = self._run_model(data)          # (M_total, 1232)
            per_point = grid_feat[global_inv]          # (B*N, 1232), cloud-major order
        return per_point.view(B, N, -1)


class SonataDecoderBackbone(nn.Module):
    """
    Decoder-probing variant (the OFFICIAL strong protocol): frozen pretrained
    encoder + a TRAINABLE PTv3 U-Net decoder (~25M, randomly initialized) that
    does learned multi-scale up-sampling with skip connections -> sharp per-point
    features at full resolution. Encoder is ALWAYS frozen (per project decision).

    Verified: instantiate PTv3 with enc_mode=False, load pretrained encoder weights
    with strict=False (all 248 missing keys are dec.*, 0 unexpected), freeze
    embedding+enc, train dec. Output feature dim = dec_channels[0] = 96.
    """
    OUT_CHANNELS = 96

    def __init__(self, ckpt: str = "sonata", repo_id: str = "facebook/sonata",
                 color_value: float = 127.5, enc_patch_size: int = 1024):
        super().__init__()
        import sonata as _sonata
        from sonata.model import PointTransformerV3
        ck = _sonata.load(ckpt, repo_id=repo_id, ckpt_only=True)
        cfg = dict(ck["config"])
        cfg.update(enc_mode=False, enable_flash=False,
                   enc_patch_size=[enc_patch_size] * 5, dec_patch_size=[enc_patch_size] * 4)
        self.model = PointTransformerV3(**cfg)
        missing, unexpected = self.model.load_state_dict(ck["state_dict"], strict=False)
        assert all(m.startswith("dec") for m in missing) and len(unexpected) == 0
        # spconv's implicit_gemm BACKWARD kernel asserts (!indices.empty()) on the
        # decoder's fine-resolution SubMConv3d — confirmed on BOTH sm_120 (5090) and
        # sm_86 (A6000). Force the Native gather-gemm-scatter algo everywhere; it is
        # slower but is the only backward that works for this decoder.
        from spconv.core import ConvAlgo
        for mod in self.model.modules():
            if isinstance(getattr(mod, "algo", None), ConvAlgo):
                mod.algo = ConvAlgo.Native

        self.color_value = color_value
        self.out_channels = self.OUT_CHANNELS
        self.fully_frozen = False   # decoder trains

        # freeze encoder (embedding + enc*), keep decoder (dec*) trainable
        for n, p in self.model.named_parameters():
            p.requires_grad = n.startswith("dec")
        self._frozen_names = [n for n, p in self.model.named_parameters() if not p.requires_grad]
        n_dec = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[SonataDecoderBackbone] decoder trainable: {n_dec/1e6:.1f}M, encoder FROZEN")
        self.model.eval()   # start eval; train() re-enables only the decoder

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()                    # frozen encoder norm stats fixed
        if mode:
            self.model.dec.train(mode)       # only the decoder trains
        return self

    def _transform_one(self, coord, normal):
        N = coord.shape[0]
        color = np.full((N, 3), self.color_value, np.float32)
        normal = np.zeros((N, 3), np.float32) if normal is None else normal.astype(np.float32)
        return sonata.transform.default()({"coord": coord.copy().astype(np.float32),
                                           "color": color, "normal": normal})

    def forward(self, coords, normals=None):
        """coords (B,N,3), normals (B,N,3)|None -> (B,N,96) per-original-point feats."""
        B, N, _ = coords.shape
        dev = next(self.model.parameters()).device
        coord_l, grid_l, color_l, feat_l, inv_l = [], [], [], [], []
        counts, cum = [], 0
        for b in range(B):
            c = coords[b].detach().cpu().numpy().astype(np.float32)
            nrm = None if normals is None else normals[b].detach().cpu().numpy()
            p = self._transform_one(c, nrm)
            m = p["coord"].shape[0]
            coord_l.append(p["coord"]); grid_l.append(p["grid_coord"])
            color_l.append(p["color"]); feat_l.append(p["feat"])
            inv_l.append(p["inverse"] + cum); counts.append(m); cum += m
        data = {"coord": torch.cat(coord_l).to(dev), "grid_coord": torch.cat(grid_l).to(dev),
                "color": torch.cat(color_l).to(dev), "feat": torch.cat(feat_l).to(dev),
                "offset": torch.tensor(np.cumsum(counts), dtype=torch.long).to(dev)}
        global_inv = torch.cat(inv_l).to(dev)
        out = self.model(data)               # enc(frozen)+dec(trainable) -> finest-res feats
        return out.feat[global_inv].view(B, N, -1)


if __name__ == "__main__":
    torch.manual_seed(0)
    bb = SonataBackbone(freeze=True).cuda()
    B, N = 2, 2048
    coords = torch.rand(B, N, 3) * 2 - 1          # ~unit sphere
    coords = coords / coords.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    out = bb(coords.cuda())
    print("backbone out:", tuple(out.shape), "expected (2,2048,1232)")
