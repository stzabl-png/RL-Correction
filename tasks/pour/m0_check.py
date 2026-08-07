"""Non-Kit M0 import and CUDA tensor check."""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-device", type=int, default=1)
    args = parser.parse_args()

    import h5py
    import isaaclab
    import tensorboardX
    import torch
    import wandb

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    device = torch.device(f"cuda:{args.cuda_device}")
    generator = torch.Generator(device=device).manual_seed(42)
    left = torch.randn(1024, 1024, generator=generator, device=device)
    right = torch.randn(1024, 1024, generator=generator, device=device)
    product = left @ right
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(product).all()):
        raise RuntimeError("CUDA matrix multiplication produced NaN/Inf")
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_device": args.cuda_device,
                "gpu": torch.cuda.get_device_name(device),
                "isaaclab": isaaclab.__file__,
                "h5py": h5py.__version__,
                "tensorboardX": tensorboardX.__version__,
                "wandb": wandb.__version__,
                "tensor_finite": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
