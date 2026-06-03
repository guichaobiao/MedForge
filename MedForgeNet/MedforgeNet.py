from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:
    raise RuntimeError("timm is required") from exc


REGISTER_COUNT = 4

SRM_KERNELS = np.array([
    [[0, 0, 0], [0, -1, 1], [0, 0, 0]], [[0, 0, 0], [1, -1, 0], [0, 0, 0]],
    [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
    [[0, 0, 1], [0, -1, 0], [0, 0, 0]], [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, -1, 0], [0, 0, 1]], [[0, 0, 0], [0, -1, 0], [1, 0, 0]],
    [[0, 1, 0], [0, -2, 0], [0, 1, 0]], [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
    [[1, 0, 0], [0, -2, 0], [0, 0, 1]], [[0, 0, 1], [0, -2, 0], [1, 0, 0]],
    [[0, 0, 0], [0, -3, 1], [0, 1, 1]], [[0, 1, 0], [0, -3, 0], [1, 0, 1]],
    [[1, 0, 1], [0, -3, 0], [0, 1, 0]], [[1, 1, 0], [1, -3, 0], [0, 0, 0]],
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
], dtype=np.float32)


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict", "net", "params"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    for prefix in ("module.", "backbone.", "teacher.", "student."):
        if any(key.startswith(prefix) for key in checkpoint.keys()):
            checkpoint = _strip_prefix(checkpoint, prefix)
            break
    return checkpoint


def _resize_pos_embed(pos_embed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pos_embed.shape == target.shape:
        return pos_embed
    if pos_embed.dim() != 3 or target.dim() != 3:
        return pos_embed
    src_n = pos_embed.shape[1]
    dst_n = target.shape[1]
    src_extra = 0
    dst_extra = 0
    src_hw = int(round(src_n ** 0.5))
    if src_hw * src_hw != src_n:
        src_hw = int(round((src_n - 1) ** 0.5))
        src_extra = 1 if src_hw * src_hw == src_n - 1 else 0
    dst_hw = int(round(dst_n ** 0.5))
    if dst_hw * dst_hw != dst_n:
        dst_hw = int(round((dst_n - 1) ** 0.5))
        dst_extra = 1 if dst_hw * dst_hw == dst_n - 1 else 0
    if src_hw * src_hw != src_n - src_extra or dst_hw * dst_hw != dst_n - dst_extra:
        return pos_embed
    extra = pos_embed[:, :src_extra] if src_extra else None
    posemb = pos_embed[:, src_extra:]
    posemb = posemb.reshape(1, src_hw, src_hw, pos_embed.shape[-1]).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=(dst_hw, dst_hw), mode="bicubic", align_corners=False)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, dst_hw * dst_hw, pos_embed.shape[-1])
    if dst_extra:
        if extra is None:
            extra = target[:, :dst_extra]
        posemb = torch.cat([extra[:, :dst_extra], posemb], dim=1)
    return posemb


class SRMBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weight = np.zeros((len(SRM_KERNELS) * 3, 3, 3, 3), dtype=np.float32)
        for idx, kernel in enumerate(SRM_KERNELS):
            for channel in range(3):
                weight[idx * 3 + channel, channel] = kernel
        self.srm = nn.Conv2d(3, len(SRM_KERNELS) * 3, kernel_size=3, padding=1, bias=False)
        self.srm.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=False)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(len(SRM_KERNELS) * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(len(SRM_KERNELS) * 3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.srm(x))


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return -grad_output


def grad_reverse(x: torch.Tensor) -> torch.Tensor:
    return _GradReverse.apply(x)


class PatchDetectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_logits = self.head(patch_tokens).squeeze(-1)
        k = max(3, patch_logits.shape[1] // 20)
        topk_vals, _ = patch_logits.topk(k, dim=1)
        patch_max_score = topk_vals.mean(dim=1)
        return patch_logits, patch_max_score


class ModalityAwareNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(384) for _ in range(4)])
        self.modality_bias = nn.Parameter(torch.zeros(4, 384))

    def forward(self, feat: torch.Tensor, modality_id: torch.Tensor) -> torch.Tensor:
        out = feat.clone()
        for m_id in range(4):
            mask = modality_id == m_id
            if mask.any():
                out[mask] = self.norms[m_id](feat[mask]) - self.modality_bias[m_id]
        unknown_mask = (modality_id < 0) | (modality_id >= 4)
        if unknown_mask.any():
            out[unknown_mask] = self.norms[0](feat[unknown_mask])
        return out


class MultiScaleViTEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch14_reg4_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=448,
        )
        self.embed_dim = self.backbone.embed_dim
        self.intermediate_layers = (3, 7, 11)
        self._intermediate_feats: Dict[int, torch.Tensor] = {}
        self._freeze()
        self._register_hooks()

    def load_checkpoint(self, path: str) -> None:
        if not Path(path).exists():
            return
        state_dict = _load_checkpoint(path)
        own_state = self.backbone.state_dict()
        if "pos_embed" in state_dict and "pos_embed" in own_state:
            if state_dict["pos_embed"].shape != own_state["pos_embed"].shape:
                state_dict["pos_embed"] = _resize_pos_embed(state_dict["pos_embed"], own_state["pos_embed"])
        if "register_tokens" in state_dict and "reg_token" in own_state:
            state_dict["reg_token"] = state_dict.pop("register_tokens")
        if "mask_token" in state_dict:
            state_dict.pop("mask_token")
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in own_state and value.shape == own_state[key].shape
        }
        self.backbone.load_state_dict(filtered, strict=False)

    def _freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        for block in self.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad_(True)
        norm = getattr(self.backbone, "norm", None)
        if norm:
            for param in norm.parameters():
                param.requires_grad_(True)

    def _register_hooks(self) -> None:
        for idx in self.intermediate_layers:
            def hook_fn(module, inputs, output, layer_idx=idx):
                self._intermediate_feats[layer_idx] = output
            self.backbone.blocks[idx].register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._intermediate_feats = {}
        feats = self.backbone.forward_features(x)
        intermediate_cls = []
        for idx in self.intermediate_layers:
            feat = self._intermediate_feats.get(idx)
            if feat is not None and feat.dim() == 3:
                intermediate_cls.append(feat[:, 0])
        return {
            "cls_token": feats[:, 0],
            "patch_tokens": feats[:, 1 + REGISTER_COUNT:],
            "intermediate_cls": intermediate_cls,
        }


class AttentionPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Linear(768, 1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.score(patch_tokens), dim=1)
        return (patch_tokens * weights).sum(dim=1)


class MedforgeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = MultiScaleViTEncoder()
        self.srm_branch = SRMBranch()
        self.attn_pool = AttentionPool()
        self.patch_head = PatchDetectionHead()
        self.fuse = nn.Sequential(
            nn.LayerNorm(3968),
            nn.Linear(3968, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 384),
        )
        self.modality_norm = ModalityAwareNorm()
        self.cls_head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Dropout(0.2),
            nn.Linear(384, 1),
        )
        self.real_domain_head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, 10),
        )

    def load_backbone_checkpoint(self, path: str) -> None:
        self.encoder.load_checkpoint(path)

    def extract_feature(
        self,
        image_448: torch.Tensor,
        image_256: torch.Tensor,
        modality_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out = self.encoder(image_448)
        cls_token = enc_out["cls_token"]
        patch_tokens = enc_out["patch_tokens"]
        intermediate_cls = enc_out["intermediate_cls"]
        pooled = self.attn_pool(patch_tokens)
        srm_feat = self.srm_branch(image_256)
        fused_in = torch.cat([cls_token, pooled] + intermediate_cls + [srm_feat], dim=-1)
        feat = self.fuse(fused_in)
        if modality_id is not None:
            feat = self.modality_norm(feat, modality_id)
        return feat, patch_tokens

    def forward(
        self,
        image_448: torch.Tensor,
        image_256: torch.Tensor,
        modality_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat, patch_tokens = self.extract_feature(image_448, image_256, modality_id)
        cls_logit = self.cls_head(feat).squeeze(-1)
        global_score = torch.sigmoid(cls_logit)
        patch_logits, patch_max_score = self.patch_head(patch_tokens)
        patch_score = torch.sigmoid(patch_max_score)
        real_domain_logit = self.real_domain_head(grad_reverse(feat))
        return {
            "cls_logit": cls_logit,
            "cls_score": global_score + 0.3 * patch_score * (1.0 - global_score),
            "global_score": global_score,
            "patch_logits": patch_logits,
            "patch_max_score": patch_max_score,
            "patch_score": patch_score,
            "feature": feat,
            "real_domain_logit": real_domain_logit,
        }


def build_MedforgeNet() -> MedforgeNet:
    return MedforgeNet()
