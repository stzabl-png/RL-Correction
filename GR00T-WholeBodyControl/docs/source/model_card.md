# Model Card

SONIC provides three released whole-body controller checkpoints for the
Unitree G1. Choose the model based on its reference representation and intended
deployment.

## Available Models

| Model | Hugging Face location | SMPL reference input | Intended use and comments |
|---|---|---|---|
| **Default SONIC (original release)** | Top-level `model_encoder.onnx`, `model_decoder.onnx`, and `observation_config.yaml`; training checkpoint at `sonic_release/last.pt` | 10 future frames at 20 ms spacing, approximately 200 ms of reference lookahead | Default general-purpose SONIC controller for motion tracking, planning, teleoperation, and compatibility with existing deployments. G1 and teleoperation future-reference observations use `step5`. |
| **Low-latency teleoperation** | [`low_latency/`](https://huggingface.co/nvidia/GEAR-SONIC/tree/main/low_latency) | 4 future frames at 20 ms spacing, approximately 80 ms of reference lookahead | Intended for more responsive whole-body teleoperation and VLA execution. G1 and teleoperation future-reference observations use `step1`. Use its encoder, decoder, and observation config together. |
| **SONIC v1.1** | [`sonic_v1_1/`](https://huggingface.co/nvidia/GEAR-SONIC/tree/main/sonic_v1_1) | 10 future frames at 20 ms spacing, approximately 200 ms of reference lookahead | Uses robot-heading-normalized target orientation and was trained with wrist-pose augmentation. Intended for heading-stable 3-point teleoperation and SONIC-backed VLA policies that use this controller. G1 and teleoperation future-reference observations use `step5`; this is not the low-latency model. |

```{image} _static/sonic_v1_1_demo.gif
:alt: SONIC v1.1 whole-body teleoperation demo
:width: 100%
:align: center
```

*SONIC v1.1: 3-point teleoperation with wrist-pose tracking, kneeling, ground
pickup, and dynamic whole-body control.*

All three models use the SONIC universal-token controller, produce 64-dimensional
latent motion tokens, run the controller at 50 Hz, and support SMPL pose, G1
motion reference, and VR 3-point inputs. Deployment uses C++ and TensorRT; the
PyTorch checkpoints support Isaac Lab evaluation and continued training.

```{note}
The lookahead values describe the reference horizon presented to the
controller. They are not measurements of total end-to-end teleoperation
latency, which also includes sensing, networking, preprocessing, and inference.
```

## Released Files

| Model | Deployment files | PyTorch and configuration files |
|---|---|---|
| Default SONIC | `model_encoder.onnx`, `model_decoder.onnx`, `observation_config.yaml` | `sonic_release/last.pt`, `sonic_release/config.yaml` |
| Low-latency teleoperation | `low_latency/model_encoder.onnx`, `low_latency/model_decoder.onnx`, `low_latency/observation_config.yaml` | `low_latency/last.pt`, `low_latency/config.yaml`, `low_latency/model_config.yaml` |
| SONIC v1.1 | `sonic_v1_1/model_encoder.onnx`, `sonic_v1_1/model_decoder.onnx`, `sonic_v1_1/observation_config.yaml` | `sonic_v1_1/last.pt`, `sonic_v1_1/config.yaml`, `sonic_v1_1/model_config.yaml` |

All files are hosted in
[`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC). Model weights
are covered by the [NVIDIA Open Model License](resources/license.md).

## Choosing a Model

Use **Default SONIC** when you want the original release, the broadest
compatibility with existing deployment setups, or the standard motion-tracking
and planning controller.

Use **Low-latency teleoperation** when responsiveness to streamed SMPL, VR, or
VLA commands is the priority. Its shorter reference horizon reduces commanded
motion lookahead, but it does not remove latency elsewhere in the system.

Use **SONIC v1.1** for robot-heading-normalized 3-point
teleoperation or a SONIC-backed VLA policy trained against this controller. It
retains the 10-frame SMPL horizon and was trained with wrist-pose augmentation.

## Usage

Install the Hugging Face dependency from the repository root:

```bash
pip install huggingface_hub
```

### Default SONIC

```bash
python download_from_hf.py

cd gear_sonic_deploy
./deploy.sh --input-type zmq_manager real
```

### Low-Latency Teleoperation

```bash
python download_from_hf.py --low-latency

cd gear_sonic_deploy
./deploy.sh \
    --cp policy/low_latency/model \
    --obs-config policy/low_latency/observation_config.yaml \
    --input-type zmq_manager \
    real
```

### SONIC v1.1

```bash
python download_from_hf.py --sonic-v1-1

cd gear_sonic_deploy
./deploy.sh \
    --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq_manager \
    real
```

### Python VLA Launcher

For the default model:

```bash
python gear_sonic/scripts/launch_inference.py \
    --camera-host 192.168.123.164 \
    --prompt "pick up the cup"
```

For the low-latency model:

```bash
python gear_sonic/scripts/launch_inference.py \
    --deploy-checkpoint policy/low_latency/model \
    --deploy-obs-config policy/low_latency/observation_config.yaml \
    --camera-host 192.168.123.164 \
    --prompt "pick up the cup"
```

For SONIC v1.1, replace the two `policy/low_latency/` paths above with
`policy/sonic_v1_1/`.

See [Downloading Model Checkpoints](getting_started/download_models.md) for
PyTorch checkpoint evaluation and additional download options.

## Limitations and Safety

- The low-latency name refers to reduced controller reference lookahead, not a
  benchmark of total system latency.
- SONIC v1.1 is not a low-latency checkpoint; it uses the
  10-frame SMPL reference horizon.
- Each ONNX encoder and decoder must be used with its matching observation
  configuration.
- These checkpoints target the Unitree G1 embodiment.
- Test in simulation before deployment and keep a safety operator ready to
  stop a physical robot.
