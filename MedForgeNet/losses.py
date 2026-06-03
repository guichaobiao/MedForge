from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: float,
    label_smoothing: float,
) -> torch.Tensor:
    if label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def patch_focal_loss(
    patch_logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    alpha: float,
    label_smoothing: float,
) -> torch.Tensor:
    patch_targets = labels.unsqueeze(1).expand_as(patch_logits)
    return focal_loss_with_logits(
        patch_logits.reshape(-1),
        patch_targets.reshape(-1),
        gamma,
        alpha,
        label_smoothing,
    )


def real_domain_loss(
    real_domain_logits: torch.Tensor,
    domain_ids: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    real_mask = domain_ids >= 0
    if not real_mask.any():
        return real_domain_logits.new_tensor(0.0), float("nan")
    logits = real_domain_logits[real_mask]
    targets = domain_ids[real_mask].long()
    loss = F.cross_entropy(logits.float(), targets)
    acc = (logits.argmax(dim=-1) == targets).float().mean().item()
    return loss, acc


def consistency_flip_448(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    for idx in range(out.shape[0]):
        if torch.rand(1, device=out.device).item() < 0.5:
            out[idx] = out[idx].flip(-1)
    return out


def feature_consistency_loss(
    model,
    image_448: torch.Tensor,
    image_256: torch.Tensor,
    modality_ids: torch.Tensor,
    feature: torch.Tensor,
    domain_ids: torch.Tensor,
) -> torch.Tensor:
    real_mask = domain_ids >= 0
    if not real_mask.any():
        return feature.new_tensor(0.0)
    aug_448 = consistency_flip_448(image_448[real_mask])
    aug_feature, _ = model.extract_feature(aug_448, image_256[real_mask], modality_ids[real_mask])
    return (1.0 - F.cosine_similarity(feature[real_mask], aug_feature, dim=-1)).mean()
