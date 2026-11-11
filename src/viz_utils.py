from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import datasets, transforms

if __package__:
    from .dataset import IMAGENET_MEAN, IMAGENET_STD, get_split_indices
    from .model import build_model
else:
    from dataset import IMAGENET_MEAN, IMAGENET_STD, get_split_indices
    from model import build_model


def denormalize(t):
    """Convert a normalized CHW image tensor to a clipped HWC array."""
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError("Expected a CHW tensor with 3 channels")

    image = t.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.clip(image * std + mean, 0.0, 1.0)


def load_model(name, arch, device):
    """Load a run's best checkpoint and return the model in evaluation mode."""
    model = build_model(arch, num_classes=10, pretrained=False).to(device)
    checkpoint_path = Path("checkpoints") / name / "best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()


def get_test_samples(seed = 42):
    """Return raw EuroSAT images and labels from the shared test split."""
    dataset = datasets.EuroSAT("data", download=True)
    _, _, test_indices = get_split_indices(dataset.targets, seed)
    return [dataset[index] for index in test_indices]


def make_transform(image_size):
    """Build the evaluation transform for a model's input resolution."""
    resize = [transforms.Resize(image_size)] if image_size != 64 else []
    return transforms.Compose(
        resize
        + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def predict_all(
    model: nn.Module,
    samples: list[tuple[Image.Image, int]],
    image_size: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict labels for every sample using the model's evaluation transform."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    transform = make_transform(image_size)
    predictions = []

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            images = torch.stack([transform(image) for image, _ in batch]).to(device)
            predictions.append(model(images).argmax(dim=1).cpu().numpy())

    if not predictions:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(predictions)


def get_hook_layers(model: nn.Module, arch: str) -> dict[str, nn.Module]:
    """Return the activation layers to visualize for a supported architecture."""
    if arch == "cnn":
        return {
            "block1": model.features[0][2],
            "block2": model.features[1][2],
            "block3": model.features[2][2],
            "block4": model.features[3][2],
        }
    if arch == "resnet18":
        return {
            "stem": model.relu,
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
        }
    raise ValueError(f"Unsupported architecture: {arch}")


def register_activation_hooks(layers):
    """Capture selected layer outputs and return a hook cleanup function."""
    activations: dict[str, torch.Tensor] = {}
    handles = []

    for name, layer in layers.items():
        def capture_output(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            layer_name: str = name,
        ) -> None:
            activations[layer_name] = output.detach().cpu()

        handles.append(layer.register_forward_hook(capture_output))

    def remove_hooks():
        for handle in handles:
            handle.remove()

    return activations, remove_hooks


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_samples = get_test_samples()
    runs = (
        ("cnn_e30", "cnn", 64),
        ("resnet18_e15", "resnet18", 128),
    )
    batch_size = 64

    print("Device:", device)
    for name, arch, image_size in runs:
        model = load_model(name, arch, device)
        transform = make_transform(image_size)
        correct = 0

        with torch.no_grad():
            for start in range(0, len(test_samples), batch_size):
                batch = test_samples[start : start + batch_size]
                images = torch.stack(
                    [transform(image) for image, _ in batch]
                ).to(device)
                labels = torch.tensor(
                    [label for _, label in batch], device=device
                )
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()

        accuracy = correct / len(test_samples)
        print(f"[{name}] test accuracy={accuracy:.3f}")