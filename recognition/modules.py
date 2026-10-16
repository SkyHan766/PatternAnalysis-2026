
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny


NUM_CLASSES = 2
CLASS_TO_INDEX = {"NC": 0, "AD": 1}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}


# The plain CNN baseline is sized to match ConvNeXt-Tiny's parameter budget.
# The final width (491) was selected to keep the trainable parameter counts
# within about 0.2% while retaining ordinary Conv-BatchNorm-ReLU blocks.
BASELINE_CHANNELS = (64, 128, 256, 512, 491)
BASELINE_DEPTHS = (2, 2, 3, 6, 6)


def set_seed(seed: int = 42) -> None:
    """Fix Python, NumPy, and PyTorch random seeds."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(require_cuda: bool = True) -> torch.device:
    """Use the Colab CUDA GPU, or CPU only when explicitly allowed."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU:", torch.cuda.get_device_name(0))
        return device

    if require_cuda:
        raise RuntimeError(
            "CUDA GPU not found. Select a GPU runtime in Google Colab."
        )

    print("Device: CPU")
    return torch.device("cpu")


class PlainCNNBaseline(nn.Module):
    """VGG-style CNN baseline using standard convolutions and max pooling."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Downsample 224x224 input to 56x56, similar to ConvNeXt's stem.
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                BASELINE_CHANNELS[0],
                kernel_size=4,
                stride=4,
                bias=False,
            ),
            nn.BatchNorm2d(BASELINE_CHANNELS[0]),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_channels = BASELINE_CHANNELS[0]

        for stage_index, (out_channels, depth) in enumerate(
            zip(BASELINE_CHANNELS, BASELINE_DEPTHS)
        ):
            blocks = []

            for block_index in range(depth):
                block_in_channels = (
                    in_channels if block_index == 0 else out_channels
                )
                blocks.extend([
                    nn.Conv2d(
                        block_in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ])

            # Ordinary max pooling performs stage-to-stage downsampling.
            if stage_index < len(BASELINE_CHANNELS) - 1:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(BASELINE_CHANNELS[-1], num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.features(outputs)
        outputs = self.pool(outputs)
        outputs = torch.flatten(outputs, start_dim=1)
        return self.classifier(outputs)


def build_model(
    model_name: str = "convnext",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """
    Build either ConvNeXt-Tiny from random initialization or the plain CNN.

    ConvNeXt is deliberately created with weights=None. No ImageNet weights
    are downloaded or loaded. Both models have two output classes: NC and AD.
    """
    model_name = model_name.lower()

    if model_name == "convnext":
        model = convnext_tiny(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name in {"baseline", "plain_cnn"}:
        return PlainCNNBaseline(num_classes=num_classes)

    raise ValueError(
        f"Unknown model_name={model_name!r}. Use 'convnext' or 'baseline'."
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def check_model_output(
    model: nn.Module,
    device: torch.device,
    input_size: int = 224,
) -> None:
    """Check that the model returns two class logits per image."""
    model.eval()
    example = torch.zeros(
        2,
        3,
        input_size,
        input_size,
        device=device,
    )

    with torch.inference_mode():
        output = model(example)

    expected_shape = (2, NUM_CLASSES)
    print("Model output shape:", tuple(output.shape))

    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Expected output shape {expected_shape}, got {tuple(output.shape)}."
        )
