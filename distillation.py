"""Core valid-region-aligned intermediate feature distillation used by RPNet-K2."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class K2FeatureDistillation(nn.Module):
    """Align RPNet Stage-3 features with the frozen teacher representation.

    RPNet receives a 640 x 640 center-padded canvas whose original content is
    480 x 640. At the /16 Stage-3 resolution, the valid content therefore
    occupies 30 of 40 rows. The five padding-only rows at each vertical edge
    are excluded before feature matching.
    """

    def __init__(
        self,
        student_channels: int = 256,
        teacher_channels: int = 128,
        feature_weight: float = 0.2,
        cosine_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.adapter = nn.Conv2d(student_channels, teacher_channels, 1, bias=True)
        self.feature_weight = float(feature_weight)
        self.cosine_weight = float(cosine_weight)

    @staticmethod
    def valid_region(student_feature: Tensor) -> Tensor:
        if student_feature.ndim != 4 or tuple(student_feature.shape[1:]) != (256, 40, 40):
            raise ValueError(
                "Expected the RPNet Stage-3 feature shape [B, 256, 40, 40], "
                f"got {tuple(student_feature.shape)}."
            )
        return student_feature[:, :, 5:35, :]

    def forward(self, student_feature: Tensor, teacher_feature: Tensor) -> Tensor:
        projected = self.adapter(self.valid_region(student_feature))
        teacher_feature = teacher_feature.detach()
        if projected.shape != teacher_feature.shape:
            raise ValueError(
                "Student and teacher features must be geometrically aligned; "
                f"got {tuple(projected.shape)} and {tuple(teacher_feature.shape)}."
            )

        mse = F.mse_loss(projected, teacher_feature)
        cosine = F.cosine_similarity(projected.flatten(1), teacher_feature.flatten(1), dim=1).mean()
        raw_feature_loss = mse + self.cosine_weight * (1.0 - cosine)
        return self.feature_weight * raw_feature_loss
