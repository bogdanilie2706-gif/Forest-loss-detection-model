import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torchvision import datasets

if __package__:
    from .model import build_model
    from .viz_utils import (
        denormalize,
        get_hook_layers,
        get_test_samples,
        load_model,
        make_transform,
        predict_all,
        register_activation_hooks,
    )
else:
    from model import build_model
    from viz_utils import (
        denormalize,
        get_hook_layers,
        get_test_samples,
        load_model,
        make_transform,
        predict_all,
        register_activation_hooks,
    )


def plot_first_layer_filters(weights, title, ncols, normalize: str = "minmax"):
    """Plot normalized RGB convolution filters in a grid."""
    if normalize not in ("centered", "minmax"):
        raise ValueError("normalize must be 'centered' or 'minmax'")
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

        filter_weights = filters[index]
        if normalize == "centered":
            maximum = filter_weights.abs().max()
            if maximum > 0:
                image = filter_weights / (2 * maximum) + 0.5
            else:
                image = torch.full_like(filter_weights, 0.5)
        else:
            minimum, maximum = filter_weights.min(), filter_weights.max()
            if maximum > minimum:
                image = (filter_weights - minimum) / (maximum - minimum)
            else:
                image = torch.zeros_like(filter_weights)
        image = image.permute(1, 2, 0)
        axis.imshow(image.numpy(), interpolation="nearest")
        axis.set_title(str(index), fontsize=8)

    figure.tight_layout()
    return figure


def select_samples(samples, cnn_preds, resnet_preds, class_names) -> list[dict]:
    """Choose representative correctly classified and CNN-missed test images."""
    selected = []
    target_classes = ("Forest", "River", "Highway", "AnnualCrop", "PermanentCrop")

    for class_name in target_classes:
        class_index = class_names.index(class_name)
        for index, (image, label) in enumerate(samples):
            if (
                label == class_index
                and cnn_preds[index] == label
                and resnet_preds[index] == label
            ):
                selected.append(
                    {
                        "index": index,
                        "role": class_name.lower(),
                        "image": image,
                        "label": label,
                        "cnn_pred": int(cnn_preds[index]),
                        "resnet_pred": int(resnet_preds[index]),
                    }
                )
                break

    wrong_index = 0
    for index, (image, label) in enumerate(samples):
        if cnn_preds[index] != label and resnet_preds[index] == label:
            wrong_index += 1
            selected.append(
                {
                    "index": index,
                    "role": f"cnn_wrong_{wrong_index}",
                    "image": image,
                    "label": label,
                    "cnn_pred": int(cnn_preds[index]),
                    "resnet_pred": int(resnet_preds[index]),
                }
            )
            if wrong_index == 2:
                break

    return selected


def plot_samples_overview(selected, class_names) -> Figure:
    """Plot selected test images with their true and predicted labels."""
    import matplotlib.pyplot as plt

    columns = 3
    rows = math.ceil(len(selected) / columns) if selected else 1
    figure, axes = plt.subplots(
        rows, columns, figsize=(columns * 4, rows * 4), squeeze=False
    )
    figure.suptitle("Selected EuroSAT test samples")

    for index, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if index >= len(selected):
            continue

        sample = selected[index]
        label = class_names[sample["label"]]
        cnn_pred = class_names[sample["cnn_pred"]]
        resnet_pred = class_names[sample["resnet_pred"]]
        axis.imshow(sample["image"])
        axis.set_title(
            f"true: {label}\nCNN: {cnn_pred}\nResNet: {resnet_pred}",
            color=(
                "red"
                if sample["cnn_pred"] != sample["label"]
                or sample["resnet_pred"] != sample["label"]
                else "black"
            ),
            fontsize=9,
            pad=4,
        )

    figure.tight_layout()
    return figure


def plot_feature_maps(
    model,
    arch: str,
    image: Image.Image,
    image_size: int,
    device: torch.device,
    title: str,
) -> Figure:
    """Plot one image and the most active feature maps at selected layers."""
    import matplotlib.pyplot as plt

    layers = get_hook_layers(model, arch)
    activations, remove_hooks = register_activation_hooks(layers)
    try:
        transform = make_transform(image_size)
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            model(input_tensor)
    finally:
        remove_hooks()

    figure, axes = plt.subplots(
        len(layers) + 1,
        9,
        figsize=(18, (len(layers) + 1) * 2.2),
        squeeze=False,
    )
    figure.suptitle(title)

    for axis in axes[0]:
        axis.set_axis_off()
    axes[0, 0].imshow(denormalize(input_tensor[0]), interpolation="nearest")
    axes[0, 0].set_title("Input")

    for row, (name, activation) in enumerate(activations.items(), start=1):
        maps = activation[0]
        channels, height, width = maps.shape
        axes[row, 0].set_ylabel(f"{name}  {channels} x {height} x {width}")
        axes[row, 0].yaxis.set_label_position("left")
        maps_to_show = [maps.mean(dim=0)]
        channel_order = maps.mean(dim=(1, 2)).argsort(descending=True)[:8]
        maps_to_show.extend(maps[channel] for channel in channel_order)

        for column, feature_map in enumerate(maps_to_show):
            minimum, maximum = feature_map.min(), feature_map.max()
            if maximum > minimum:
                feature_map = (feature_map - minimum) / (maximum - minimum)
            else:
                feature_map = torch.zeros_like(feature_map)
            axes[row, column].imshow(
                feature_map.numpy(), interpolation="nearest", cmap="viridis"
            )

        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

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
    parser.add_argument(
        "command", choices=["first-layer", "samples", "feature-maps"]
    )
    args = parser.parse_args()

    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "first-layer":
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
    else:
        samples = get_test_samples()
        class_names = datasets.EuroSAT("data").classes
        cnn_model = load_model("cnn_e30", "cnn", device)
        resnet_model = load_model("resnet18_e15", "resnet18", device)
        cnn_preds = predict_all(cnn_model, samples, 64, device)
        resnet_preds = predict_all(resnet_model, samples, 128, device)
        selected = select_samples(samples, cnn_preds, resnet_preds, class_names)

        if args.command == "samples":
            figure = plot_samples_overview(selected, class_names)
            path = output_dir / "samples_overview.png"
            figure.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved {path}")
        else:
            selected_by_role = {sample["role"]: sample for sample in selected}
            for role in ("forest", "highway", "cnn_wrong_1"):
                sample = selected_by_role.get(role)
                if sample is None:
                    continue
                for model, arch, run_name, image_size in (
                    (cnn_model, "cnn", "cnn_e30", 64),
                    (resnet_model, "resnet18", "resnet18_e15", 128),
                ):
                    figure = plot_feature_maps(
                        model,
                        arch,
                        sample["image"],
                        image_size,
                        device,
                        f"{run_name} feature maps: {role}",
                    )
                    path = output_dir / f"feature_maps_{run_name}_{role}.png"
                    figure.savefig(path, dpi=100, bbox_inches="tight")
                    plt.close(figure)
                    print(f"Saved {path}")