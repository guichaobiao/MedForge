from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline import DataPipelineConfig, build_eval_loader, build_train_loader, load_data_pipeline_config
from losses import (
    feature_consistency_loss,
    focal_loss_with_logits,
    patch_focal_loss,
    real_domain_loss,
)
from MedforgeNet import build_MedforgeNet
from metrics import compute_metrics
from train_utils import (
    amp_autocast,
    build_scaler,
    load_resume,
    resolve_path,
    save_checkpoint,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--test_chatgpt_manifest")
    parser.add_argument("--test_gemini_manifest")
    parser.add_argument("--backbone_checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--head_lr", type=float, required=True)
    parser.add_argument("--backbone_lr", type=float, required=True)
    parser.add_argument("--srm_lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--lambda_bce", type=float, required=True)
    parser.add_argument("--lambda_patch", type=float, required=True)
    parser.add_argument("--lambda_domain", type=float, required=True)
    parser.add_argument("--lambda_consistency", type=float, required=True)
    parser.add_argument("--focal_gamma", type=float, required=True)
    parser.add_argument("--focal_alpha", type=float, required=True)
    parser.add_argument("--label_smoothing", type=float, required=True)
    parser.add_argument("--grad_clip", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--max_train_batches", type=int)
    parser.add_argument("--max_eval_batches", type=int)
    return parser.parse_args()


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> AdamW:
    backbone_params = [
        param
        for name, param in model.named_parameters()
        if "encoder.backbone" in name and param.requires_grad
    ]
    srm_params = [
        param
        for name, param in model.named_parameters()
        if "srm_branch" in name and param.requires_grad
    ]
    head_params = [
        param
        for name, param in model.named_parameters()
        if "encoder.backbone" not in name and "srm_branch" not in name and param.requires_grad
    ]
    return AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": srm_params, "lr": args.srm_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    data_config: DataPipelineConfig,
    max_batches: int = 0,
) -> Dict[str, Union[float, int]]:
    model.eval()
    labels_all = []
    scores_all = []
    for batch_idx, batch in enumerate(loader):
        image_large = batch[data_config.output_image_large_key].to(device, non_blocking=True)
        image_small = batch[data_config.output_image_small_key].to(device, non_blocking=True)
        modality_ids = batch[data_config.output_modality_key].to(device, non_blocking=True)
        output = model(image_large, image_small, modality_id=modality_ids)
        labels_all.append(batch[data_config.output_label_key].cpu().numpy().astype(np.int64))
        scores_all.append(output["cls_score"].detach().float().cpu().numpy())
        if max_batches and batch_idx + 1 >= max_batches:
            break
    labels = np.concatenate(labels_all) if labels_all else np.array([], dtype=np.int64)
    scores = np.concatenate(scores_all) if scores_all else np.array([], dtype=np.float32)
    return compute_metrics(labels, scores)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    args: argparse.Namespace,
    data_config: DataPipelineConfig,
) -> Dict[str, float]:
    model.train()
    amp_enabled = bool(args.amp) and device.type == "cuda"
    meters: Dict[str, list[float]] = {
        "loss_total": [],
        "loss_bce": [],
        "loss_patch": [],
        "loss_domain": [],
        "loss_consistency": [],
        "mean_score_real": [],
        "mean_score_fake": [],
        "domain_acc": [],
    }
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        image_large = batch[data_config.output_image_large_key].to(device, non_blocking=True)
        image_small = batch[data_config.output_image_small_key].to(device, non_blocking=True)
        labels = batch[data_config.output_label_key].to(device, non_blocking=True).float()
        domain_ids = batch[data_config.output_domain_key].to(device, non_blocking=True)
        modality_ids = batch[data_config.output_modality_key].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(device, amp_enabled):
            output = model(image_large, image_small, modality_id=modality_ids)
            loss_bce = focal_loss_with_logits(
                output["cls_logit"],
                labels,
                args.focal_gamma,
                args.focal_alpha,
                args.label_smoothing,
            )
            loss_patch = patch_focal_loss(
                output["patch_logits"],
                labels,
                args.focal_gamma,
                args.focal_alpha,
                args.label_smoothing,
            )
            if args.lambda_domain > 0:
                loss_domain, domain_acc = real_domain_loss(output["real_domain_logit"], domain_ids)
            else:
                loss_domain = output["cls_logit"].new_tensor(0.0)
                domain_acc = float("nan")
            if args.lambda_consistency > 0:
                loss_consistency = feature_consistency_loss(
                    model,
                    image_large,
                    image_small,
                    modality_ids,
                    output["feature"],
                    domain_ids,
                )
            else:
                loss_consistency = output["cls_logit"].new_tensor(0.0)
            loss = (
                args.lambda_bce * loss_bce
                + args.lambda_patch * loss_patch
                + args.lambda_domain * loss_domain
                + args.lambda_consistency * loss_consistency
            )
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scores = output["cls_score"].detach().float().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        meters["loss_total"].append(float(loss.detach().cpu()))
        meters["loss_bce"].append(float(loss_bce.detach().cpu()))
        meters["loss_patch"].append(float(loss_patch.detach().cpu()))
        meters["loss_domain"].append(float(loss_domain.detach().cpu()))
        meters["loss_consistency"].append(float(loss_consistency.detach().cpu()))
        if (labels_np == 0).any():
            meters["mean_score_real"].append(float(scores[labels_np == 0].mean()))
        if (labels_np == 1).any():
            meters["mean_score_fake"].append(float(scores[labels_np == 1].mean()))
        if not math.isnan(domain_acc):
            meters["domain_acc"].append(float(domain_acc))
        if args.max_train_batches and batch_idx + 1 >= args.max_train_batches:
            break
    result = {key: float(np.mean(value)) if value else float("nan") for key, value in meters.items()}
    result["epoch_time"] = time.time() - start_time
    return result


def format_metrics(prefix: str, metrics: Dict[str, Any]) -> str:
    parts = []
    for key in ["ap", "acc_fake", "acc_real", "avg_acc", "n", "mean_score_real", "mean_score_fake"]:
        value = metrics.get(key)
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        elif value is not None:
            parts.append(f"{key}={value}")
    return prefix + " " + " ".join(parts)


def format_log(epoch: int, train_metrics: Dict[str, float], eval_metrics: Dict[str, Dict[str, Any]]) -> str:
    train_part = (
        f"epoch={epoch} "
        f"loss={train_metrics['loss_total']:.4f} "
        f"bce={train_metrics['loss_bce']:.4f} "
        f"patch={train_metrics['loss_patch']:.4f} "
        f"domain={train_metrics['loss_domain']:.4f} "
        f"consistency={train_metrics['loss_consistency']:.4f} "
        f"domain_acc={train_metrics['domain_acc']:.4f} "
        f"time={train_metrics['epoch_time']:.0f}s"
    )
    eval_part = " | ".join(format_metrics(name, values) for name, values in eval_metrics.items())
    return train_part + " | " + eval_part


def build_eval_loaders(args: argparse.Namespace, root: Path, data_config: DataPipelineConfig) -> Dict[str, Any]:
    loaders: Dict[str, Any] = {}
    val_loader, _ = build_eval_loader(resolve_path(args.val_manifest, root), args.batch_size, args.num_workers, data_config)
    loaders["val"] = val_loader
    if args.test_chatgpt_manifest:
        loader, _ = build_eval_loader(resolve_path(args.test_chatgpt_manifest, root), args.batch_size, args.num_workers, data_config)
        loaders["chatgpt"] = loader
    if args.test_gemini_manifest:
        loader, _ = build_eval_loader(resolve_path(args.test_gemini_manifest, root), args.batch_size, args.num_workers, data_config)
        loaders["gemini"] = loader
    return loaders


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = load_data_pipeline_config(resolve_path(args.data_config, PROJECT_ROOT))
    train_manifest = resolve_path(args.train_manifest, PROJECT_ROOT)
    train_loader, train_dataset = build_train_loader(train_manifest, args.batch_size, args.num_workers, data_config)
    eval_loaders = build_eval_loaders(args, PROJECT_ROOT, data_config)
    model = build_MedforgeNet().to(device)
    if args.backbone_checkpoint:
        model.load_backbone_checkpoint(str(resolve_path(args.backbone_checkpoint, PROJECT_ROOT)))
    optimizer = make_optimizer(model, args)
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_resume(resolve_path(args.resume, PROJECT_ROOT), model, optimizer, device)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = build_scaler(device, bool(args.amp) and device.type == "cuda")
    run_dir = resolve_path(args.output_dir, PROJECT_ROOT) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"train={len(train_dataset)} " + " ".join(f"{name}={len(loader.dataset)}" for name, loader in eval_loaders.items()))
    best_score = -1.0
    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args, data_config)
        scheduler.step()
        eval_metrics = {
            name: evaluate(model, loader, device, data_config, args.max_eval_batches)
            for name, loader in eval_loaders.items()
        }
        log_line = format_log(epoch, train_metrics, eval_metrics)
        print(log_line, flush=True)
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            epoch,
            eval_metrics,
            vars(args),
        )
        main_metric = float(eval_metrics["val"].get("avg_acc", float("nan")))
        if not math.isnan(main_metric) and main_metric > best_score:
            best_score = main_metric
            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                epoch,
                eval_metrics,
                vars(args),
            )
        write_json(run_dir / "history.json", history)
    print(f"done {run_dir}")


if __name__ == "__main__":
    main()
