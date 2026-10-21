import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # randare în fișiere, fără fereastră (potrivit pentru WSL)
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

from dataset import get_dataloaders
from model import SimpleCNN


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return {
        "loss": total_loss / len(labels),
        "acc": (preds == labels).float().mean().item(),
        "macro_f1": f1_score(labels.numpy(), preds.numpy(), average="macro"),
        "preds": preds,
        "labels": labels,
    }


def save_confusion_matrix(res, classes, path):
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay.from_predictions(
        res["labels"].numpy(), res["preds"].numpy(),
        display_labels=classes, xticks_rotation=45, ax=ax, colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_curves(history_path, out_path):
    history = json.loads(Path(history_path).read_text())
    epochs = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(epochs, [h["train_loss"] for h in history], label="train")
    a1.plot(epochs, [h["val_loss"] for h in history], label="val")
    a1.set_title("Loss"); a1.set_xlabel("epocă"); a1.legend()
    a2.plot(epochs, [h["train_acc"] for h in history], label="train")
    a2.plot(epochs, [h["val_acc"] for h in history], label="val")
    a2.set_title("Acuratețe"); a2.set_xlabel("epocă"); a2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(name, split):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_loader, test_loader, classes = get_dataloaders(num_workers=0)
    loader = {"val": val_loader, "test": test_loader}[split]

    ckpt_dir = Path("checkpoints") / name
    out_dir = Path("results") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SimpleCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))

    res = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    report = classification_report(res["labels"].numpy(), res["preds"].numpy(),
                                   target_names=classes, digits=3)
    summary = (f"[{name} | {split}] loss={res['loss']:.3f} "
               f"acc={res['acc']:.3f} macro_f1={res['macro_f1']:.3f}")
    print(summary)
    print(report)

    (out_dir / f"report_{split}.txt").write_text(summary + "\n\n" + report)
    save_confusion_matrix(res, classes, out_dir / f"confusion_matrix_{split}.png")

    history_path = ckpt_dir / "history.json"
    if history_path.exists():
        save_curves(history_path, out_dir / "training_curves.png")

    print(f"Salvat în {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="numele rulării, ex. cnn_e15")
    p.add_argument("--split", choices=["val", "test"], default="val")
    args = p.parse_args()
    main(args.name, args.split)