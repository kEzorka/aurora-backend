"""Model lifecycle: load the checkpoint once, keep it resident, roll it forward."""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

import torch
from aurora import Batch, rollout

from . import config

# V100 is compute capability 7.0: fp16 tensor cores yes, bf16 no. "off" keeps
# everything in fp32, which is the safe default for a first prototype.
AUTOCAST = os.environ.get("AURORA_AUTOCAST", "off")

# torch.compile pays a one-off warmup to trace and fuse the graph. Worth it for
# a long rollout, not for a one-step job. Off by default for that reason.
COMPILE = os.environ.get("AURORA_COMPILE", "0") == "1"


def _model_class(name: str):
    import aurora

    try:
        return getattr(aurora, name)
    except AttributeError:
        available = [n for n in dir(aurora) if n.startswith("Aurora")]
        raise ValueError(f"unknown model {name!r}; aurora exposes {available}") from None


class AuroraEngine:
    def __init__(
        self,
        device: str | None = None,
        model_name: str | None = None,
        autocast: str | None = None,
        compile_model: bool | None = None,
    ):
        self.device = device or config.DEVICE
        self.model_name = model_name or config.MODEL_NAME
        # Taken as arguments and not only from the environment so that a
        # benchmark can hold everything else fixed and vary one of them.
        self.autocast = autocast or AUTOCAST
        self.compile_model = COMPILE if compile_model is None else compile_model

        # Moving the model is not enough to pin the process to one GPU: any
        # tensor created as plain "cuda" — inside the library, inside autocast,
        # inside torch.cuda.max_memory_allocated() — still goes to the default
        # device. Without this, AURORA_DEVICE=cuda:2 quietly touches cuda:0 too.
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        self.model = _model_class(self.model_name)()
        self.model.load_checkpoint()
        self.model.eval()

        if config.ACTIVATION_CHECKPOINTING and hasattr(
            self.model, "configure_activation_checkpointing"
        ):
            self.model.configure_activation_checkpointing()

        self.model = self.model.to(self.device)

        if self.compile_model:
            self.model = torch.compile(self.model)

    def _autocast(self):
        if self.autocast == "fp16":
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def rollout(self, batch: Batch, steps: int) -> Iterator[Batch]:
        """Yield `steps` predictions, each 6 h further out, already on the CPU.

        Every prediction is moved off the GPU as it arrives — holding all of
        them in VRAM is what blows up a long rollout, not the forward pass.
        """
        batch = batch.to(self.device)
        with torch.inference_mode(), self._autocast():
            for pred in rollout(self.model, batch, steps=steps):
                yield pred.to("cpu")

    def unload(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
