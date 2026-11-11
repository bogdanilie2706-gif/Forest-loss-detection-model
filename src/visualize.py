import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.figure import Figure

if __package__:
    from .model import build_model
    from .viz_utils import load_model
else:
    from model import build_model
    from viz_utils import load_model


def plot_first_layer_filters(weights, title, ncols):
    """Plot independently normalized RGB convolution filters in a grid."""
    if weights.ndim != 4 or weights.shape[1] != 3:
        raise ValueError("Expected filter weights with shape [N, 3, k, k]")
    if ncols < 1:
        raise ValueError("ncols must be positive")

    import matplotlib.pyplot as plt

    filters = weights.detach().cpu().float()
    rows = math.ceil(filters.shape[0] / ncols)
    figure, axes = plt.subplots(
        rows, ncols, figsize=(ncols * 1.5, rows * 1.5), squeeze=False
    )
    figure.suptitle(title)

    for index, axis in enumerate(axes.flat):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        if index >= filters.shape[0]:
            continue

        image = filters[index].permute(1, 2, 0)
        minimum, maximum = image.min(), image.max()
        if maximum > minimum:
            image = (image - minimum) / (maximum - minimum)
        else:
            image = torch.zeros_like(image)
        axis.imshow(image.numpy(), interpolation="nearest")
        axis.set_title(str(index), fontsize=8)

    figure.tight_layout()
    return figure


def plot_resnet_filter_change():
    """Plot relative conv1 filter changes from ImageNet pretraining."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetuned_model = load_model("resnet18_e15", "resnet18", device)
    imagenet_model = build_model("resnet18", pretrained=True).to(device)

    finetuned_weights = finetuned_model.conv1.weight.detach()
    imagenet_weights = imagenet_model.conv1.weight.detach()
    changes = (
        (finetuned_weights - imagenet_weights).flatten(start_dim=1).norm(dim=1)
        / imagenet_weights.flatten(start_dim=1).norm(dim=1)
    ).cpu().numpy()

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(np.arange(len(changes)), changes)
    axis.set_xlabel("Filter index")
    axis.set_ylabel("Relative change (||fine-tuned - ImageNet|| / ||ImageNet||)")
    axis.set_title("ResNet18 conv1 filter change")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Visualize trained model filters.")
    parser.add_argument("command", choices=["first-layer"])
    args = parser.parse_args()

    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_model = load_model("cnn_e30", "cnn", device)
    cnn_figure = plot_first_layer_filters(
        cnn_model.features[0][0].weight,
        title="SimpleCNN first-layer filters",
        ncols=8,
    )
    cnn_path = output_dir / "first_layer_cnn_e30.png"
    cnn_figure.savefig(cnn_path, dpi=100, bbox_inches="tight")
    plt.close(cnn_figure)
    print(f"Saved {cnn_path}")

    resnet_model = load_model("resnet18_e15", "resnet18", device)
    resnet_figure = plot_first_layer_filters(
        resnet_model.conv1.weight,
        title="ResNet18 first-layer filters",
        ncols=8,
    )
    resnet_path = output_dir / "first_layer_resnet18_e15.png"
    resnet_figure.savefig(resnet_path, dpi=100, bbox_inches="tight")
    plt.close(resnet_figure)
    print(f"Saved {resnet_path}")

    change_figure = plot_resnet_filter_change()
    change_path = output_dir / "resnet_filter_change.png"
    change_figure.savefig(change_path, dpi=100, bbox_inches="tight")
    plt.close(change_figure)
    print(f"Saved {change_path}")