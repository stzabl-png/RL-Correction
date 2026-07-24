# Environment setup

Each step uses its own conda environment because the upstream models depend on different CUDA/PyTorch stacks. The old `pipeline/` wrappers are not required for the active reconstruction path.

## Submodule init

```bash
git submodule update --init \
  third_party/vipe third_party/sam3 third_party/hawor \
  third_party/sam2 third_party/sam-3d-objects third_party/FoundationPose
```

## Per-step environments

| Step | Env (typical) | Reference |
|------|---------------|-----------|
| vipe | `cu128` + `cd third_party/vipe && uv run python …` | `third_party/vipe/docs/installation.md` |
| sam3_hands | `sam3` | `third_party/sam3/README.md` |
| sam2_object | `sam3` (includes editable `third_party/sam2`) | `third_party/sam2/INSTALL.md` |
| hawor | `hawor` | `third_party/hawor` upstream setup plus MANO files |
| sam3d | SAM3D env (PyTorch3D, kaolin) | `third_party/sam-3d-objects/README.md` |
| sam3d_scale | FoundationPose/SAM3D-compatible env with `scipy` | `third_party/FoundationPose/readme.md`, `third_party/sam-3d-objects/README.md` |
| fp_pose | `foundationpose` (see below) | `third_party/FoundationPose/readme.md` |
| fuse | `hawor` or lightweight (`numpy`, `opencv`, `trimesh`, `joblib`) | — |

Run each `run_sequence.py` with the appropriate env activated.

ViPE Python packages live in `third_party/vipe/.venv` (managed by **uv**), not in conda directly. Conda `cu128` supplies CUDA tooling and `uv`; always invoke:

```bash
conda activate cu128
cd third_party/vipe
uv run python ../../recon_pipeline/vipe/run_sequence.py --dataset ... --video-id ... --video ... --gpu 0
```

## SAM2 object masks

SAM2 is used for manual object mask preview and video propagation in `recon_pipeline/sam2_object/`. This repo uses the existing `sam3` conda environment for SAM2 as well; `third_party/sam2` is installed editable inside `sam3`. SAM3 remains the hand-mask backend in `recon_pipeline/sam3_hands/`; SAM3D is a separate mesh reconstruction dependency.

`sam3_hands` requests `sam3.1` by default. If the assigned CUDA device is a
Blackwell GPU such as RTX 5090, the runner automatically falls back to `sam3`
for that step because `sam3.1` can hit floating-point precision failures on
that architecture. Existing non-Blackwell runs still use `sam3.1`.

Verify the integrated env:

```bash
conda activate sam3
python -c "from sam2.build_sam import build_sam2_video_predictor; print('sam2 ok')"
python -c "import torch, sam2._C; print('sam2 extension ok')"
python -m pip show SAM-2
```

If SAM2 is missing from `sam3`, install it into that same environment and build the optional CUDA extension. The extension provides SAM2 mask post-processing; without it, SAM2 prints a warning and skips hole filling.

```bash
conda activate sam3
cd third_party/sam2
SAM2_BUILD_ALLOW_ERRORS=0 uv pip install -e .
cd checkpoints
./download_ckpts.sh
```

If `nvcc` is not on `PATH` but the CUDA 12.8 compiler is available from the repo's `cu128` env, build the extension in place with:

```bash
conda activate sam3
cd third_party/sam2
CUDA_HOME=/home/jiakaichen/miniconda3/envs/cu128 \
PATH=/home/jiakaichen/miniconda3/envs/cu128/bin:$PATH \
SAM2_BUILD_ALLOW_ERRORS=0 \
python setup.py build_ext --inplace
```

The default object runner expects:

```text
third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```

Pass `--sam2-checkpoint` and `--sam2-model-cfg` to use another SAM2.1 model size.

## FoundationPose (`fp_pose`)

```bash
conda create -n foundationpose python=3.10 -y
conda activate foundationpose
cd third_party/FoundationPose
# Follow official readme: install PyTorch, nvdiffrast, weights under weights/
python -m pip install uv
uv pip install -r requirements.txt  # if present; else follow readme pip steps
```

Download FoundationPose weights per [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose).

Verify:

```bash
conda activate foundationpose
python -c "import sys; sys.path.insert(0,'third_party/FoundationPose'); from estimater import FoundationPose; print('ok')"
```

## SAM3D checkpoints

Download SAM3D checkpoints into `third_party/sam-3d-objects/checkpoints/hf/` per upstream README before running `sam3d/run_sequence.py`.

`sam3d/run_sequence.py` produces the raw mesh. `sam3d_scale/run_sequence.py` then loads that raw mesh, the SAM2 object mask, ViPE depth, and FoundationPose helpers to produce `object_mesh_scaled_final.obj`.

## Isaac ROS FP++ (STEP_6_pose only)

Not used by `recon_pipeline`. See [fp_pose/ISAAC_ROS.md](../fp_pose/ISAAC_ROS.md) if you need the legacy TensorRT path on branch `STEP_6_pose`.

## Depth scale

ViPE depth may not be metric. Tune `--depth-scale` on `sam3d_scale/run_sequence.py` before the object mesh is scaled, then keep the same depth scale when running `fp_pose/run_sequence.py` if you override its default. Start with `1.0`, then adjust using known object size or HOI4D GT if available.

If `sam3d_scale` or `fp_pose` fails on import with `ModuleNotFoundError: scipy`, install SciPy in the active FoundationPose-compatible environment. If `fuse` fails on import with `ModuleNotFoundError: joblib`, run it in the HaWoR environment or install `joblib` in the lightweight visualization environment.
