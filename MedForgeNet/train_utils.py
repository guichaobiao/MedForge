from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

try:
    from torch.amp import GradScaler, autocast as torch_autocast

    def amp_autocast(device: torch.device, enabled: bool):
        return torch_autocast(device_type=device.type, enabled=enabled)

    def build_scaler(device: torch.device, enabled: bool):
        return GradScaler(device.type, enabled=enabled)

except ImportError:
    from torch.cuda.amp import GradScaler, autocast as torch_autocast

    def amp_autocast(device: torch.device, enabled: bool):
        return torch_autocast(enabled=enabled)

    def build_scaler(device: torch.device, enabled: bool):
        return GradScaler(enabled=enabled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: Union[str, Path], root: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return root / value


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    args: Dict[str, Any],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": args,
        },
        output_path,
    )


def load_resume(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, None] = None,
    device: Union[torch.device, None] = None,
) -> Tuple[int, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0, checkpoint


def write_json(path: Union[str, Path], value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(value, handle, indent=2)
