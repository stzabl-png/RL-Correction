# Task 5 VM environment

The Task-5 environment is isolated from existing lab environments:

```text
account: kailang@128.32.164.89
conda: /home/kailang/.local/miniconda3/envs/rl-correction-pour
Isaac Sim: 5.1.0
Isaac Lab: v2.3.2 (37ddf62)
PyTorch: 2.7.0+cu128
Python: 3.11
```

Installed project-facing packages include `h5py 3.16.0`,
`tensorboardX 2.6.5`, and `wandb 0.19.11`. Runtime compatibility pins retained
from the Isaac Sim kernel are `typing_extensions 4.12.2`, `psutil 5.9.8`,
`starlette 0.45.3`, `click 8.1.7`, `daqp 0.7.2`, and `flatdict 4.0.1`.

The non-Kit check passed on GPU1 with a finite CUDA tensor:

```bash
CUDA_VISIBLE_DEVICES=1 \
  /home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python \
  -m tasks.pour.m0_check --cuda-device 0
```

Before every Isaac launch, check GPU occupancy and use the empty GPU. The first
Kit launch is intentionally blocked until the account owner accepts the NVIDIA
Omniverse EULA; this repository does not automate acceptance of legal terms.

The upstream environment has three known metadata conflicts that do not affect
the passed non-Kit check: ONNX requests a newer `typing_extensions`, IPython
requests a newer `psutil`, and Isaac Lab's declared Starlette version conflicts
with the Isaac Sim kernel's FastAPI/Starlette pair. Do not upgrade these pins
without a separate compatibility test.
