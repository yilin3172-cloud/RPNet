"""Standalone inference implementation of RPNet.

Paper-facing modules:
    DualModalShallowFeatureEncoder
    CHRISStem: Cross-Modal Hierarchical Residual Infusion Stem
    P3Block
    AMF
    DRSRHead: Detail-Residual Semantic Restoration Head

The model accepts a six-channel tensor containing RGB followed by the
three-channel thermal representation and returns semantic and boundary logits.
Training-only teacher networks, distillation adapters, losses, and data-loading
logic are intentionally excluded.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PathLike = Union[str, os.PathLike[str]]

DEFAULT_THERMAL_MEAN = 0.39541009068489075
DEFAULT_THERMAL_STD = 0.18480300903320312


def autopad(k: Union[int, Tuple[int, int]], p: Optional[Union[int, Tuple[int, int]]] = None, d: int = 1):
    """Return padding that preserves the spatial shape for a stride-one convolution."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else tuple(d * (value - 1) + 1 for value in k)
    if p is None:
        p = k // 2 if isinstance(k, int) else tuple(value // 2 for value in k)
    return p


class Conv(nn.Module):
    """Convolution, batch normalization, and SiLU activation used by RPNet."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: Union[int, Tuple[int, int]] = 1,
        s: Union[int, Tuple[int, int]] = 1,
        p: Optional[Union[int, Tuple[int, int]]] = None,
        g: int = 1,
        d: int = 1,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


def _load_frontend_features(weights_path: Optional[PathLike]) -> nn.Sequential:
    """Create the MobileNetV2 hierarchy used by the shallow and context encoders.

    A local ImageNet checkpoint can be supplied when constructing a model for
    training. For inference, random initialization is sufficient because a
    complete RPNet checkpoint replaces every parameter and running statistic.
    """
    try:
        from torchvision.models import mobilenet_v2
    except ImportError as error:
        raise RuntimeError("RPNet requires torchvision to construct its pretrained frontend.") from error

    model = mobilenet_v2(weights=None)
    if weights_path is not None:
        path = Path(weights_path).expanduser().resolve()
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location="cpu")
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(f"Expected a mapping checkpoint at {path}, got {type(state_dict).__name__}.")
        state_dict = state_dict.get("state_dict", state_dict)
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(f"The state_dict entry at {path} is not a mapping.")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "The supplied pretrained frontend checkpoint does not match the expected feature hierarchy."
            ) from error
    return model.features


def _reset_batchnorm_running_stats(module: nn.Module) -> None:
    """Reset running statistics for the thermal branch while preserving learned affine parameters."""
    for child in module.modules():
        if isinstance(child, nn.BatchNorm2d):
            child.running_mean.zero_()
            child.running_var.fill_(1.0)
            child.num_batches_tracked.zero_()


def _apply_deployment_module_settings(module: nn.Module) -> None:
    """Apply the non-state-dict BatchNorm and activation settings of the source model builder."""
    for child in module.modules():
        if type(child) is nn.BatchNorm2d:
            child.eps = 1e-3
            child.momentum = 0.03
        elif type(child) in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            child.inplace = True


class DualModalShallowFeatureEncoder(nn.Module):
    """Independent RGB and thermal shallow encoding followed by channel concatenation.

    Input:  B x 6 x H x W
    Output: D_shallow, B x 32 x H/2 x W/2
    """

    def __init__(
        self,
        c1: int = 6,
        c2: int = 32,
        *,
        frontend_weights_path: Optional[PathLike] = None,
        thermal_mean: float = DEFAULT_THERMAL_MEAN,
        thermal_std: float = DEFAULT_THERMAL_STD,
    ):
        super().__init__()
        if c1 != 6 or c2 != 32:
            raise ValueError("DualModalShallowFeatureEncoder requires c1=6 and c2=32.")
        if not 0.0 <= thermal_mean <= 1.0 or not 0.0 < thermal_std <= 1.0:
            raise ValueError("Thermal normalization values must satisfy 0 <= mean <= 1 and 0 < std <= 1.")

        features = _load_frontend_features(frontend_weights_path)
        self.rgb_encoder = nn.Sequential(copy.deepcopy(features[0]), copy.deepcopy(features[1]))
        self.thermal_encoder = nn.Sequential(copy.deepcopy(features[0]), copy.deepcopy(features[1]))
        _reset_batchnorm_running_stats(self.thermal_encoder)

        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("thermal_mean", torch.tensor([thermal_mean]).view(1, 1, 1, 1))
        self.register_buffer("thermal_std", torch.tensor([thermal_std]).view(1, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"DualModalShallowFeatureEncoder expects [B, 6, H, W], got {tuple(x.shape)}.")
        rgb = (x[:, :3] - self.rgb_mean) / self.rgb_std
        thermal = (x[:, 3:6] - self.thermal_mean) / self.thermal_std
        return torch.cat((self.rgb_encoder(rgb), self.thermal_encoder(thermal)), dim=1)


class CHRISStem(nn.Module):
    """Cross-Modal Hierarchical Residual Infusion Stem.

    D_shallow is unpacked into its RGB and thermal feature groups. The two
    modalities first form a base at H/4 x W/4, then add aligned H/8 x W/8
    branch-local context through a learnable residual gain initialized to 0.25.

    Input:  B x 32 x H/2 x W/2
    Output: B x 64 x H/4 x W/4
    """

    def __init__(self, c1: int = 32, c2: int = 64, *, frontend_weights_path: Optional[PathLike] = None):
        super().__init__()
        if c1 != 32 or c2 != 64:
            raise ValueError("CHRISStem requires c1=32 and c2=64.")

        features = _load_frontend_features(frontend_weights_path)
        self.rgb_encoder4 = copy.deepcopy(features[2])
        self.thermal_encoder4 = copy.deepcopy(features[2])
        self.rgb_encoder8 = nn.Sequential(copy.deepcopy(features[3]), copy.deepcopy(features[4]))
        self.thermal_encoder8 = nn.Sequential(copy.deepcopy(features[3]), copy.deepcopy(features[4]))
        _reset_batchnorm_running_stats(self.thermal_encoder4)
        _reset_batchnorm_running_stats(self.thermal_encoder8)

        self.rgb_proj = nn.Conv2d(24, c2, 1, bias=False)
        self.thermal_proj = nn.Conv2d(24, c2, 1, bias=False)
        self.context_rgb_proj = nn.Conv2d(32, c2, 1, bias=False)
        self.context_thermal_proj = nn.Conv2d(32, c2, 1, bias=False)
        self.context_norm = nn.BatchNorm2d(c2)
        self.fuse_norm = nn.BatchNorm2d(c2)
        self.fuse_act = nn.SiLU()
        self.context_gain_logit = nn.Parameter(torch.tensor(-1.0986123))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 32:
            raise ValueError(f"CHRISStem expects [B, 32, H/2, W/2], got {tuple(x.shape)}.")
        rgb2, thermal2 = torch.split(x, 16, dim=1)
        rgb4 = self.rgb_encoder4(rgb2)
        thermal4 = self.thermal_encoder4(thermal2)
        base = self.rgb_proj(rgb4) + self.thermal_proj(thermal4)

        context = self.context_rgb_proj(self.rgb_encoder8(rgb4))
        context = context + self.context_thermal_proj(self.thermal_encoder8(thermal4))
        context = self.context_norm(context)
        context = F.interpolate(context, size=base.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse_act(self.fuse_norm(base + self.context_gain_logit.sigmoid() * context))


class MultiKernelDWConv(nn.Module):
    """Parallel 3x3, 5x5, and 7x7 depthwise ripple convolution."""

    def __init__(self, dim: int, kernel_sizes: Tuple[int, int, int] = (3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim, bias=False) for kernel_size in kernel_sizes]
        )
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(sum(conv(x) for conv in self.convs) / len(self.convs) + x)


class MRFFILight(nn.Module):
    """Multi-receptive-field ripple refinement used by the active RPNet graph."""

    def __init__(self, dim: int, use_wavelet: bool = False):
        super().__init__()
        if use_wavelet:
            raise ValueError("The delivered RPNet architecture uses ripple refinement without a wavelet path.")
        self.use_wavelet = False
        self.mkconv = MultiKernelDWConv(dim)
        # This unused branch is retained because it exists in the delivered checkpoint topology.
        self.fuse = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.mkconv(x) + x


class P3Block(nn.Module):
    """Local depthwise perception, ripple refinement, four-times FFN, and residual connection."""

    def __init__(self, c1: int, c2: int, stride: int = 1, use_wavelet: bool = False):
        super().__init__()
        self.stride = stride
        channels = c2 if stride > 1 else c1
        self.downsample = nn.Conv2d(c1, c2, 3, stride, 1, bias=False) if stride > 1 else None
        self.norm1 = nn.BatchNorm2d(channels)
        self.local_conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mrffi = MRFFILight(channels, use_wavelet=use_wavelet)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, 4 * channels, 1, bias=False),
            nn.BatchNorm2d(4 * channels),
            nn.SiLU(),
            nn.Conv2d(4 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            x = self.downsample(x)
            identity = x
        x = self.act(self.norm1(x))
        x = self.act(self.norm2(self.local_conv(x)))
        x = self.mrffi(x)
        return self.ffn(x) + identity


class AMF(nn.Module):
    """Additive multi-scale fusion with ripple refinement.

    Inputs:
        F3: B x 64 x H/4 x W/4
        F4: B x 128 x H/8 x W/8
        F5: B x 256 x H/16 x W/16
    Output:
        M: B x 128 x H/4 x W/4
    """

    def __init__(self, in_chs: Tuple[int, int, int] = (64, 128, 256), out_ch: int = 128):
        super().__init__()
        if tuple(in_chs) != (64, 128, 256) or out_ch != 128:
            raise ValueError("The delivered RPNet AMF requires in_chs=(64, 128, 256) and out_ch=128.")
        self.reduce_convs = nn.ModuleList([Conv(channels, out_ch, 1) for channels in in_chs])
        self.mrffi = MRFFILight(out_ch, use_wavelet=False)
        self.final_conv = Conv(out_ch, out_ch, 1)

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("AMF expects [F3, F4, F5].")
        p3, p4, p5 = (module(feature) for module, feature in zip(self.reduce_convs, x))
        target_size = p3.shape[-2:]
        fused = p3 + F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        fused = fused + F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)
        return self.final_conv(self.mrffi(fused))


class DRSRHead(nn.Module):
    """Detail-Residual Semantic Restoration Head.

    The semantic branch is projected and aligned to D_shallow. The shallow
    detail branch then provides bounded residual compensation before separate
    semantic and boundary classifiers are bilinearly restored to input scale.
    """

    def __init__(self, nc: int = 9, ncb: int = 1, ch: Tuple[int, int] = (32, 128), hidden_ch: int = 64):
        super().__init__()
        if tuple(ch) != (32, 128) or hidden_ch != 64:
            raise ValueError("DRSRHead requires ch=(32, 128) and hidden_ch=64 in the delivered RPNet.")
        self.nc = nc
        self.ncb = ncb
        detail_ch, semantic_ch = ch
        self.main_proj = Conv(semantic_ch, hidden_ch, 1, 1, act=True)
        self.detail_proj = nn.Sequential(
            Conv(detail_ch, detail_ch, 3, 1, g=detail_ch, act=True),
            Conv(detail_ch, hidden_ch, 1, 1, act=True),
        )
        self.refine = nn.Sequential(
            Conv(hidden_ch, hidden_ch, 3, 1, g=hidden_ch, act=True),
            Conv(hidden_ch, hidden_ch, 1, 1, act=True),
        )
        self.beta_logit = nn.Parameter(torch.tensor(-1.0986123))
        self.seg_conv = nn.Conv2d(hidden_ch, nc, 1, bias=True)
        self.boundary_conv = nn.Conv2d(hidden_ch, ncb, 1, bias=True)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("DRSRHead expects [D_shallow, M].")
        detail, semantic = x
        main = self.main_proj(semantic)
        main = F.interpolate(main, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        fused = main + self.beta_logit.sigmoid() * self.detail_proj(detail)
        fused = self.refine(fused)
        output_size = (detail.shape[-2] * 2, detail.shape[-1] * 2)
        semantic_logits = F.interpolate(self.seg_conv(fused), size=output_size, mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(
            self.boundary_conv(fused), size=output_size, mode="bilinear", align_corners=False
        )
        return semantic_logits, boundary_logits


class RPNet(nn.Module):
    """RPNet inference network with a 6-channel RGB-thermal input.

    For a B x 6 x 640 x 640 input, the forward route is:
        D_shallow: B x 32 x 320 x 320
        F3:        B x 64 x 160 x 160
        F4:        B x 128 x 80 x 80
        F5:        B x 256 x 40 x 40
        M:         B x 128 x 160 x 160
        outputs:   B x 9 x 640 x 640 and B x 1 x 640 x 640
    """

    def __init__(
        self,
        nc: int = 9,
        ncb: int = 1,
        *,
        frontend_weights_path: Optional[PathLike] = None,
        thermal_mean: float = DEFAULT_THERMAL_MEAN,
        thermal_std: float = DEFAULT_THERMAL_STD,
    ):
        super().__init__()
        if nc != 9 or ncb != 1:
            raise ValueError("The delivered RPNet output contract requires nc=9 and ncb=1.")

        self.model = nn.ModuleList(
            [
                DualModalShallowFeatureEncoder(
                    6,
                    32,
                    frontend_weights_path=frontend_weights_path,
                    thermal_mean=thermal_mean,
                    thermal_std=thermal_std,
                ),
                CHRISStem(32, 64, frontend_weights_path=frontend_weights_path),
                nn.Sequential(P3Block(64, 64, 1, False), P3Block(64, 64, 1, False)),
                Conv(64, 128, 3, 2),
                nn.Sequential(P3Block(128, 128, 1, False), P3Block(128, 128, 1, False)),
                Conv(128, 256, 3, 2),
                nn.Sequential(P3Block(256, 256, 1, False), P3Block(256, 256, 1, False)),
                AMF((64, 128, 256), 128),
                DRSRHead(nc, ncb, (32, 128), 64),
            ]
        )
        # These values are not part of a state_dict but are required to reproduce source-model inference.
        _apply_deployment_module_settings(self)

    def forward(self, x: Tensor, *, return_stage3: bool = False):
        detail = self.model[0](x)
        f_mid = self.model[1](detail)
        f3 = self.model[2](f_mid)
        f4 = self.model[4](self.model[3](f3))
        f5 = self.model[6](self.model[5](f4))
        semantic = self.model[7]((f3, f4, f5))
        semantic_logits, boundary_logits = self.model[8]((detail, semantic))
        if return_stage3:
            return semantic_logits, boundary_logits, f5
        return semantic_logits, boundary_logits


def _load_checkpoint_payload(checkpoint_path: PathLike) -> Mapping[str, object]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RPNet checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Expected a checkpoint mapping at {path}, got {type(payload).__name__}.")
    return payload


def load_rpnet_checkpoint(
    model: RPNet,
    checkpoint_path: PathLike,
    *,
    strict: bool = True,
) -> nn.modules.module._IncompatibleKeys:
    """Load a delivered student checkpoint into RPNet.

    The release checkpoints store the inference weights under
    ``model_state_dict``. ``strict=True`` is recommended because the standalone
    model intentionally preserves that state-dict topology.
    """
    payload = _load_checkpoint_payload(checkpoint_path)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("The checkpoint model_state_dict is not a mapping.")
    return model.load_state_dict(state_dict, strict=strict)


def parameter_count(model: nn.Module) -> int:
    """Return the number of parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    network = RPNet().eval()
    print(f"RPNet parameters: {parameter_count(network):,}")
    with torch.inference_mode():
        semantic_output, boundary_output = network(torch.randn(1, 6, 64, 64))
    print(f"Semantic logits: {tuple(semantic_output.shape)}")
    print(f"Boundary logits: {tuple(boundary_output.shape)}")
