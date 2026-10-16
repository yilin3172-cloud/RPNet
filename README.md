# RPNet: A Ripple-based Progressive Network for RGB-T Semantic Segmentation

This repository provides the core implementation of RPNet. It contains the
inference architecture and the valid-region-aligned intermediate feature
distillation used by RPNet-K2. Training pipelines, datasets, experiment logs,
ablation files, and checkpoints are intentionally excluded.

## Files

```text
.
|-- rpnet.py          # Complete RPNet architecture
|-- distillation.py   # Core RPNet-K2 feature-distillation module
|-- requirements.txt
`-- LICENSE
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Model

```python
import torch
from rpnet import RPNet, load_rpnet_checkpoint

model = RPNet()
load_rpnet_checkpoint(model, "rpnet_base.pt")
model.eval()

rgbt = torch.rand(1, 6, 640, 640)
with torch.inference_mode():
    semantic_logits, boundary_logits = model(rgbt)
```

The six input channels are ordered as RGB followed by the thermal image
replicated across three channels. Inputs are expected in `[0, 1]`; modality-
specific normalization is performed inside the model.

For ImageNet initialization during training, pass the torchvision MobileNetV2
checkpoint through `frontend_weights_path` when constructing `RPNet`.

## RPNet-K2 distillation

```python
from distillation import K2FeatureDistillation

semantic_logits, boundary_logits, student_stage3 = model(
    rgbt, return_stage3=True
)
teacher_stage3 = frozen_teacher_feature  # [B, 128, 30, 40]

distiller = K2FeatureDistillation()
loss = supervised_loss + distiller(student_stage3, teacher_stage3)
```

The distillation module removes padding-only rows from the student's
`40 x 40` Stage-3 feature map, projects its 256 channels to 128 channels, and
matches the resulting `30 x 40` valid region to the frozen teacher feature.
The teacher and adapter are used only during training.

## Checkpoints and citation

Checkpoint download links and citation metadata will be added after the
manuscript and release artifacts are finalized.

## License

This code is released under the GNU AGPL-3.0 license. The RPNet implementation
retains compatibility with the modified Ultralytics-based training code used
during development. MobileNetV2 blocks are constructed with torchvision.
