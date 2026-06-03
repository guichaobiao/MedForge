from __future__ import annotations

import csv
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


@dataclass
class DataPipelineConfig:
    image_path_column: str
    label_column: str
    real_subsource_column: str
    dataset_source_column: str
    dataset_column: str
    output_image_large_key: str
    output_image_small_key: str
    output_label_key: str
    output_domain_key: str
    output_dataset_source_key: str
    output_modality_key: str
    output_path_key: str
    image_mode: str
    fallback_size: Tuple[int, int]
    fallback_color: Tuple[int, int, int]
    normalize_mean: List[float]
    normalize_std: List[float]
    fake_label: int
    fake_domain_id: int
    unknown_real_subsource_id: int
    unknown_modality_id: int
    real_subsource_ids: Dict[str, int]
    modality_ids: Dict[str, int]
    train_large_resize: int
    train_large_crop: int
    train_large_crop_scale: Tuple[float, float]
    train_large_crop_ratio: Tuple[float, float]
    train_small_resize: int
    train_hflip_p: float
    train_vflip_p: float
    color_jitter_brightness: float
    color_jitter_contrast: float
    color_jitter_saturation: float
    color_jitter_hue: float
    jpeg_p: float
    jpeg_qmin: int
    jpeg_qmax: int
    jpeg_format: str
    eval_large_resize: int
    eval_large_crop: int
    eval_small_resize: int
    eval_small_crop: int
    pin_memory: bool
    train_drop_last: bool
    eval_drop_last: bool
    sampler_replacement: bool
    filter_missing_files: bool

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "DataPipelineConfig":
        columns = values["columns"]
        outputs = values["outputs"]
        image = values["image"]
        normalization = values["normalization"]
        metadata = values["metadata"]
        train = values["transforms"]["train"]
        evaluation = values["transforms"]["eval"]
        color_jitter = train["color_jitter"]
        jpeg = train["jpeg"]
        loader = values["loader"]
        return cls(
            image_path_column=columns["image_path"],
            label_column=columns["label"],
            real_subsource_column=columns["real_subsource"],
            dataset_source_column=columns["dataset_source"],
            dataset_column=columns["dataset"],
            output_image_large_key=outputs["image_large"],
            output_image_small_key=outputs["image_small"],
            output_label_key=outputs["label"],
            output_domain_key=outputs["domain_id"],
            output_dataset_source_key=outputs["dataset_source"],
            output_modality_key=outputs["modality_id"],
            output_path_key=outputs["image_path"],
            image_mode=image["mode"],
            fallback_size=tuple(image["fallback_size"]),
            fallback_color=tuple(image["fallback_color"]),
            normalize_mean=list(normalization["mean"]),
            normalize_std=list(normalization["std"]),
            fake_label=int(metadata["fake_label"]),
            fake_domain_id=int(metadata["fake_domain_id"]),
            unknown_real_subsource_id=int(metadata["unknown_real_subsource_id"]),
            unknown_modality_id=int(metadata["unknown_modality_id"]),
            real_subsource_ids={str(key): int(value) for key, value in metadata["real_subsource_ids"].items()},
            modality_ids={str(key): int(value) for key, value in metadata["modality_ids"].items()},
            train_large_resize=int(train["large_resize"]),
            train_large_crop=int(train["large_crop"]),
            train_large_crop_scale=tuple(train["large_crop_scale"]),
            train_large_crop_ratio=tuple(train["large_crop_ratio"]),
            train_small_resize=int(train["small_resize"]),
            train_hflip_p=float(train["hflip_p"]),
            train_vflip_p=float(train["vflip_p"]),
            color_jitter_brightness=float(color_jitter["brightness"]),
            color_jitter_contrast=float(color_jitter["contrast"]),
            color_jitter_saturation=float(color_jitter["saturation"]),
            color_jitter_hue=float(color_jitter["hue"]),
            jpeg_p=float(jpeg["p"]),
            jpeg_qmin=int(jpeg["qmin"]),
            jpeg_qmax=int(jpeg["qmax"]),
            jpeg_format=jpeg["format"],
            eval_large_resize=int(evaluation["large_resize"]),
            eval_large_crop=int(evaluation["large_crop"]),
            eval_small_resize=int(evaluation["small_resize"]),
            eval_small_crop=int(evaluation["small_crop"]),
            pin_memory=bool(loader["pin_memory"]),
            train_drop_last=bool(loader["train_drop_last"]),
            eval_drop_last=bool(loader["eval_drop_last"]),
            sampler_replacement=bool(loader["sampler_replacement"]),
            filter_missing_files=bool(loader["filter_missing_files"]),
        )


def load_data_pipeline_config(path: Union[str, Path]) -> DataPipelineConfig:
    with Path(path).open() as handle:
        return DataPipelineConfig.from_dict(json.load(handle))


def real_subsource_id(tag: str, label: int, config: DataPipelineConfig) -> int:
    if label == config.fake_label:
        return config.fake_domain_id
    return config.real_subsource_ids.get(tag, config.unknown_real_subsource_id)


def modality_id(dataset_name: str, config: DataPipelineConfig) -> int:
    return config.modality_ids.get(dataset_name, config.unknown_modality_id)


class RandomJPEG:
    def __init__(self, p: float, qmin: int, qmax: int, image_format: str, image_mode: str) -> None:
        self.p = p
        self.qmin = qmin
        self.qmax = qmax
        self.image_format = image_format
        self.image_mode = image_mode

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format=self.image_format, quality=random.randint(self.qmin, self.qmax))
        buffer.seek(0)
        return Image.open(buffer).convert(self.image_mode)


def tensor_normalize(config: DataPipelineConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.normalize_mean, config.normalize_std),
    ])


class MedforgeNetDataset(Dataset):
    def __init__(self, manifest_path: Union[str, Path], config: DataPipelineConfig, train: bool) -> None:
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.train = train
        self.rows: List[Dict[str, str]] = []
        if not self.manifest_path.exists():
            raise FileNotFoundError(str(self.manifest_path))
        with self.manifest_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = row.get(config.image_path_column)
                label = row.get(config.label_column)
                has_image = image_path is not None and Path(image_path).exists()
                if image_path is not None and label is not None and (has_image or not config.filter_missing_files):
                    self.rows.append(row)
        self._to_tensor = tensor_normalize(config)
        if train:
            self._resize_large = transforms.Resize(config.train_large_resize)
            self._crop_large = transforms.RandomResizedCrop(
                config.train_large_crop,
                scale=config.train_large_crop_scale,
                ratio=config.train_large_crop_ratio,
            )
            self._resize_small = transforms.Resize(config.train_small_resize)
            self._color_jitter = transforms.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=config.color_jitter_saturation,
                hue=config.color_jitter_hue,
            )
            self._jpeg = RandomJPEG(
                config.jpeg_p,
                config.jpeg_qmin,
                config.jpeg_qmax,
                config.jpeg_format,
                config.image_mode,
            )
        else:
            self._eval_large = transforms.Compose([
                transforms.Resize(config.eval_large_resize),
                transforms.CenterCrop(config.eval_large_crop),
            ])
            self._eval_small = transforms.Compose([
                transforms.Resize(config.eval_small_resize),
                transforms.CenterCrop(config.eval_small_crop),
            ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        image_path = row[self.config.image_path_column]
        image = self._read_image(image_path)
        label = int(row[self.config.label_column])
        if self.train:
            image_large, image_small = self._train_transform(image)
        else:
            image_large, image_small = self._eval_transform(image)
        return {
            self.config.output_image_large_key: image_large,
            self.config.output_image_small_key: image_small,
            self.config.output_label_key: label,
            self.config.output_domain_key: real_subsource_id(row.get(self.config.real_subsource_column, ""), label, self.config),
            self.config.output_dataset_source_key: row.get(self.config.dataset_source_column, ""),
            self.config.output_modality_key: modality_id(row.get(self.config.dataset_column, ""), self.config),
            self.config.output_path_key: row.get(self.config.image_path_column, ""),
        }

    def _read_image(self, image_path: str) -> Image.Image:
        try:
            return Image.open(image_path).convert(self.config.image_mode)
        except Exception:
            return Image.new(self.config.image_mode, self.config.fallback_size, self.config.fallback_color)

    def _train_transform(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image_large = self._crop_large(self._resize_large(image))
        image_small = self._resize_small(image)
        if random.random() < self.config.train_hflip_p:
            image_large = TF.hflip(image_large)
            image_small = TF.hflip(image_small)
        if random.random() < self.config.train_vflip_p:
            image_large = TF.vflip(image_large)
            image_small = TF.vflip(image_small)
        image_large = self._jpeg(self._color_jitter(image_large))
        return self._to_tensor(image_large), self._to_tensor(image_small)

    def _eval_transform(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._to_tensor(self._eval_large(image)), self._to_tensor(self._eval_small(image))

    @property
    def labels(self) -> np.ndarray:
        return np.array([int(row[self.config.label_column]) for row in self.rows], dtype=np.int64)


def build_balanced_sampler(dataset: MedforgeNetDataset) -> WeightedRandomSampler:
    labels = dataset.labels
    counts = np.bincount(labels).astype(np.float64)
    weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(
        torch.from_numpy(weights).double(),
        len(dataset),
        replacement=dataset.config.sampler_replacement,
    )


def build_train_loader(
    manifest: Union[str, Path],
    batch_size: int,
    num_workers: int,
    config: DataPipelineConfig,
) -> Tuple[DataLoader, MedforgeNetDataset]:
    dataset = MedforgeNetDataset(manifest, config, train=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=build_balanced_sampler(dataset),
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.train_drop_last,
    )
    return loader, dataset


def build_eval_loader(
    manifest: Union[str, Path],
    batch_size: int,
    num_workers: int,
    config: DataPipelineConfig,
) -> Tuple[DataLoader, MedforgeNetDataset]:
    dataset = MedforgeNetDataset(manifest, config, train=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.eval_drop_last,
    )
    return loader, dataset
