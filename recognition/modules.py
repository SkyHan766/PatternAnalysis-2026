
import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    convnext_tiny,
)


NUM_CLASSES = 2
CLASS_TO_INDEX = {"NC": 0, "AD": 1}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}


def set_seed(seed: int = 42) -> None:
    """固定 Python、PyTorch 和 CUDA 的随机种子。"""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(require_cuda: bool = True) -> torch.device:
    """
    选择计算设备。

    notebook 要求在 Colab GPU 上训练，因此默认要求 CUDA。
    如果你确实要在 CPU 上进行小规模调试，可以传入 require_cuda=False。
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU:", torch.cuda.get_device_name(0))
        return device

    if require_cuda:
        raise RuntimeError(
            "当前没有检测到 CUDA GPU。请在 Colab 中选择 GPU 运行时。"
        )

    device = torch.device("cpu")
    print("Device: CPU")
    return device


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    创建 ConvNeXt-Tiny 二分类模型。

    pretrained=True:
        使用 TorchVision 的 ImageNet1K_V1 预训练权重。

    pretrained=False:
        不下载预训练权重，适用于从本项目 checkpoint 加载模型。
    """
    weights = (
        ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )

    model = convnext_tiny(weights=weights)

    # ConvNeXt-Tiny 原分类器输出 1000 个 ImageNet 类别。
    # 替换最后一层，使其输出 NC 和 AD 两类。
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


def check_model_output(
    model: nn.Module,
    device: torch.device,
    input_size: int = 224,
) -> None:
    """用虚拟输入检查模型输出形状是否为 [batch_size, 2]。"""
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
            f"模型输出应为 {expected_shape}，"
            f"实际得到 {tuple(output.shape)}。"
        )
