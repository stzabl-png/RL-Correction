# TensorBoard writer that ALSO mirrors every add_scalar to Weights & Biases.
# Enabled by DEFAULT (the rebuild previously logged to TensorBoard only -> nothing showed on W&B).
# Disable with SHARPA_WANDB=0. Configure with:
#   SHARPA_WANDB_PROJECT (default "sharpa-rl-rebuild"), SHARPA_WANDB_ENTITY (default: your default entity),
#   SHARPA_WANDB_MODE (default "online"; "offline"/"disabled" also work).
import math
import os
from tensorboardX import SummaryWriter


class TBWriter:
    def __init__(self, logdir, config=None, run_name=None):
        self.writer = SummaryWriter(logdir)
        self._wandb = None
        if os.environ.get("SHARPA_WANDB", "1") == "0":
            return
        try:
            import wandb
            run_dir = os.path.dirname(os.path.normpath(logdir))           # .../<run>/stage*_tb -> .../<run>
            stage = os.path.basename(os.path.normpath(logdir)).replace("_tb", "")
            name = os.environ.get("SHARPA_WANDB_NAME") or run_name or f"{os.path.basename(run_dir)}_{stage}"
            cfg = None
            try:
                cfg = dict(config) if config is not None else None
            except Exception:
                cfg = None
            self._wandb = wandb.init(
                project=os.environ.get("SHARPA_WANDB_PROJECT", "sharpa-rl-rebuild"),
                entity=os.environ.get("SHARPA_WANDB_ENTITY") or None,
                name=name,
                dir=run_dir or ".",
                config=cfg,
                mode=os.environ.get("SHARPA_WANDB_MODE", "online"),
            )
            print(f"[wandb] ON -> project={self._wandb.project} name={name} "
                  f"(disable with SHARPA_WANDB=0)")
        except Exception as e:
            print(f"[wandb] disabled ({type(e).__name__}: {e}) -> TensorBoard only")
            self._wandb = None

    def add_scalar(self, tag, value, step=None):
        # tensorboardX otherwise emits an anonymous x2num warning and writes an
        # unusable event value.  Name the offending metric and keep it out of
        # both logging backends; this does not affect the optimizer state.
        try:
            scalar = float(value.detach().item()) if hasattr(value, "detach") else float(value)
        except (TypeError, ValueError, RuntimeError):
            scalar = None
        if scalar is not None and not math.isfinite(scalar):
            print(f"[metrics] skip non-finite scalar: {tag}={scalar}", flush=True)
            return
        self.writer.add_scalar(tag, value, step)
        if self._wandb is not None:
            try:
                import wandb
                wandb.log({tag: float(value)}, step=int(step) if step is not None else None)
            except Exception:
                pass

    def add_histogram(self, *a, **k):
        return self.writer.add_histogram(*a, **k)

    def flush(self):
        try:
            self.writer.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.writer.close()
        finally:
            if self._wandb is not None:
                try:
                    import wandb
                    wandb.finish()
                except Exception:
                    pass
