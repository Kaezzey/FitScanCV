from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:
    raise ImportError("torch is required to use scripts.models.bodym_models.") from exc

ModelVariant = Literal["single_view", "dual_view_late_fusion"]
SingleViewName = Literal["mask", "mask_left"]
BackboneName = Literal["light_cnn", "resnet18"]

VALID_MODEL_VARIANTS: tuple[str, ...] = ("single_view", "dual_view_late_fusion")
VALID_SINGLE_VIEW_NAMES: tuple[str, ...] = ("mask", "mask_left")
VALID_BACKBONE_NAMES: tuple[str, ...] = ("light_cnn", "resnet18")
GENDER_TO_VALUE: dict[str, float] = {
    "female": 0.0,
    "male": 1.0,
}


@dataclass(frozen=True)
class BodyMModelConfig:
    """Configuration for baseline and stronger BodyM regression models."""

    variant: ModelVariant = "single_view"
    num_targets: int = 14
    single_view_name: SingleViewName = "mask"
    backbone_name: BackboneName = "light_cnn"
    pretrained: bool = False
    image_embedding_dim: int = 128
    metadata_embedding_dim: int = 32
    hidden_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.variant not in VALID_MODEL_VARIANTS:
            raise ValueError(
                f"variant must be one of {VALID_MODEL_VARIANTS}, received: {self.variant!r}"
            )
        if self.single_view_name not in VALID_SINGLE_VIEW_NAMES:
            raise ValueError(
                f"single_view_name must be one of {VALID_SINGLE_VIEW_NAMES}, "
                f"received: {self.single_view_name!r}"
            )
        if self.backbone_name not in VALID_BACKBONE_NAMES:
            raise ValueError(
                f"backbone_name must be one of {VALID_BACKBONE_NAMES}, "
                f"received: {self.backbone_name!r}"
            )
        if self.pretrained and self.backbone_name != "resnet18":
            raise ValueError("pretrained=True is only supported for backbone_name='resnet18'.")
        if self.num_targets <= 0:
            raise ValueError("num_targets must be a positive integer.")
        if self.image_embedding_dim <= 0:
            raise ValueError("image_embedding_dim must be a positive integer.")
        if self.metadata_embedding_dim <= 0:
            raise ValueError("metadata_embedding_dim must be a positive integer.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")


def _validate_view_tensor(view: Tensor, name: str) -> Tensor:
    if not isinstance(view, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if view.ndim != 4:
        raise ValueError(f"{name} must have shape [B, 1, H, W].")
    if view.shape[1] != 1:
        raise ValueError(f"{name} must be grayscale with channel dimension 1.")
    return view.to(dtype=torch.float32)


def _coerce_numeric_batch(
    value: Any,
    batch_size: int,
    device: torch.device,
    name: str,
) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.reshape(-1)

    if tensor.numel() != batch_size:
        raise ValueError(
            f"{name} must contain {batch_size} value(s), received {tensor.numel()}."
        )
    return tensor


def _encode_gender_batch(
    value: Any,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if isinstance(value, str):
        gender_values = [value]
    elif isinstance(value, (list, tuple)):
        gender_values = [str(item) for item in value]
    else:
        raise TypeError(
            "metadata['hwg_gender'] must be a string or a list/tuple of strings."
        )

    if len(gender_values) != batch_size:
        raise ValueError(
            f"metadata['hwg_gender'] must contain {batch_size} value(s), "
            f"received {len(gender_values)}."
        )

    encoded_values: list[float] = []
    for item in gender_values:
        normalized = str(item).strip().lower()
        if normalized not in GENDER_TO_VALUE:
            raise ValueError(
                f"Unknown hwg_gender value: {item!r}. Expected 'female' or 'male'."
            )
        encoded_values.append(GENDER_TO_VALUE[normalized])

    return torch.tensor(encoded_values, dtype=torch.float32, device=device)


def _metadata_to_tensor(
    metadata: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    required_keys = ("hwg_gender", "hwg_height_cm", "hwg_weight_kg")
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:
        raise ValueError(f"metadata is missing required keys: {missing_keys}")

    gender = _encode_gender_batch(metadata["hwg_gender"], batch_size, device)
    height = _coerce_numeric_batch(
        metadata["hwg_height_cm"],
        batch_size=batch_size,
        device=device,
        name="metadata['hwg_height_cm']",
    )
    weight = _coerce_numeric_batch(
        metadata["hwg_weight_kg"],
        batch_size=batch_size,
        device=device,
        name="metadata['hwg_weight_kg']",
    )
    return torch.stack((gender, height, weight), dim=1)


class _GrayscaleCnnEncoder(nn.Module):
    """Lightweight grayscale CNN encoder for silhouette masks."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, view: Tensor) -> Tensor:
        encoded = self.features(view)
        return self.projection(encoded)


def _create_resnet18_backbone(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to use backbone_name='resnet18'."
        ) from exc

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        return resnet18(weights=weights)
    except Exception as exc:  # pragma: no cover - defensive wrapper for download/runtime issues
        raise RuntimeError(
            "Failed to initialize torchvision resnet18. "
            "If pretrained=True, ensure the weights are available or internet access is enabled."
        ) from exc


def _adapt_resnet18_first_conv(backbone: nn.Module, pretrained: bool) -> None:
    old_conv = getattr(backbone, "conv1", None)
    if not isinstance(old_conv, nn.Conv2d):
        raise TypeError("Expected a backbone with a Conv2d 'conv1' layer.")

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        if pretrained:
            # Preserve the activation scale that the pretrained backbone expects:
            # duplicating one grayscale mask into three RGB channels is equivalent
            # to summing the pretrained RGB kernels across the input-channel axis.
            new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)

    backbone.conv1 = new_conv


class _ResNet18Encoder(nn.Module):
    """ResNet18-based grayscale encoder for stronger paired-view modeling."""

    def __init__(self, embedding_dim: int, pretrained: bool) -> None:
        super().__init__()
        backbone = _create_resnet18_backbone(pretrained=pretrained)
        _adapt_resnet18_first_conv(backbone, pretrained=pretrained)
        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, view: Tensor) -> Tensor:
        encoded = self.backbone(view)
        return self.projection(encoded)


def _build_image_encoder(
    backbone_name: BackboneName,
    embedding_dim: int,
    pretrained: bool,
) -> nn.Module:
    if backbone_name == "light_cnn":
        return _GrayscaleCnnEncoder(embedding_dim)
    if backbone_name == "resnet18":
        return _ResNet18Encoder(embedding_dim=embedding_dim, pretrained=pretrained)
    raise ValueError(
        f"Unsupported backbone_name: {backbone_name!r}. Expected one of {VALID_BACKBONE_NAMES}."
    )


class _MetadataEncoder(nn.Module):
    """Small MLP for HWG metadata fusion."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, metadata: dict[str, Any], batch_size: int, device: torch.device) -> Tensor:
        metadata_tensor = _metadata_to_tensor(metadata, batch_size=batch_size, device=device)
        return self.network(metadata_tensor)


class _RegressionHead(nn.Module):
    """Simple MLP head for multi-target regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_targets: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class SingleViewBodyMRegressor(nn.Module):
    """Single-view regressor with configurable image backbone and HWG fusion."""

    def __init__(self, config: BodyMModelConfig, initialize_pretrained: bool | None = None) -> None:
        super().__init__()
        if config.variant != "single_view":
            raise ValueError(
                "SingleViewBodyMRegressor requires config.variant='single_view'."
            )

        self.config = config
        self.single_view_name = config.single_view_name
        self.image_encoder = _build_image_encoder(
            backbone_name=config.backbone_name,
            embedding_dim=config.image_embedding_dim,
            pretrained=config.pretrained if initialize_pretrained is None else initialize_pretrained,
        )
        self.metadata_encoder = _MetadataEncoder(config.metadata_embedding_dim)
        self.regression_head = _RegressionHead(
            input_dim=config.image_embedding_dim + config.metadata_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_targets=config.num_targets,
            dropout=config.dropout,
        )

    def forward(self, view: Tensor, metadata: dict[str, Any]) -> Tensor:
        view_tensor = _validate_view_tensor(view, self.single_view_name)
        batch_size = int(view_tensor.shape[0])
        image_embedding = self.image_encoder(view_tensor)
        metadata_embedding = self.metadata_encoder(
            metadata,
            batch_size=batch_size,
            device=view_tensor.device,
        )
        fused = torch.cat((image_embedding, metadata_embedding), dim=1)
        return self.regression_head(fused)


class DualViewLateFusionBodyMRegressor(nn.Module):
    """Late-fusion regressor over paired mask and mask_left views."""

    def __init__(self, config: BodyMModelConfig, initialize_pretrained: bool | None = None) -> None:
        super().__init__()
        if config.variant != "dual_view_late_fusion":
            raise ValueError(
                "DualViewLateFusionBodyMRegressor requires "
                "config.variant='dual_view_late_fusion'."
            )

        self.config = config
        self.image_encoder = _build_image_encoder(
            backbone_name=config.backbone_name,
            embedding_dim=config.image_embedding_dim,
            pretrained=config.pretrained if initialize_pretrained is None else initialize_pretrained,
        )
        self.metadata_encoder = _MetadataEncoder(config.metadata_embedding_dim)
        self.regression_head = _RegressionHead(
            input_dim=(config.image_embedding_dim * 2) + config.metadata_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_targets=config.num_targets,
            dropout=config.dropout,
        )

    def forward(self, mask: Tensor, mask_left: Tensor, metadata: dict[str, Any]) -> Tensor:
        mask_tensor = _validate_view_tensor(mask, "mask")
        mask_left_tensor = _validate_view_tensor(mask_left, "mask_left")
        if mask_tensor.shape[0] != mask_left_tensor.shape[0]:
            raise ValueError("mask and mask_left must have the same batch dimension.")

        batch_size = int(mask_tensor.shape[0])
        mask_embedding = self.image_encoder(mask_tensor)
        mask_left_embedding = self.image_encoder(mask_left_tensor)
        metadata_embedding = self.metadata_encoder(
            metadata,
            batch_size=batch_size,
            device=mask_tensor.device,
        )
        fused = torch.cat((mask_embedding, mask_left_embedding, metadata_embedding), dim=1)
        return self.regression_head(fused)


def build_bodym_model(
    config: BodyMModelConfig,
    initialize_pretrained: bool | None = None,
) -> nn.Module:
    if config.variant == "single_view":
        return SingleViewBodyMRegressor(config, initialize_pretrained=initialize_pretrained)
    if config.variant == "dual_view_late_fusion":
        return DualViewLateFusionBodyMRegressor(
            config,
            initialize_pretrained=initialize_pretrained,
        )
    raise ValueError(
        f"Unsupported model variant: {config.variant!r}. "
        f"Expected one of {VALID_MODEL_VARIANTS}."
    )


def build_regression_loss(name: str = "smooth_l1") -> nn.Module:
    normalized_name = name.strip().lower()
    if normalized_name == "smooth_l1":
        return nn.SmoothL1Loss(reduction="mean")
    if normalized_name == "mse":
        return nn.MSELoss(reduction="mean")
    raise ValueError("Loss name must be 'smooth_l1' or 'mse'.")


__all__ = [
    "BackboneName",
    "BodyMModelConfig",
    "DualViewLateFusionBodyMRegressor",
    "ModelVariant",
    "SingleViewBodyMRegressor",
    "build_bodym_model",
    "build_regression_loss",
]
