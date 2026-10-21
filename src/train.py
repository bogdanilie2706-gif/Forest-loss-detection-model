import json
from pathlib import Path

import torch
from torch import nn

from dataset import get_dataloaders
from model import SimpleCNN
from evaluate import evaluate
import argparse


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


def main(epochs=15, lr=1e-3, weight_decay=1e-4, seed=42, name="cnn"):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader, _, classes = get_dataloaders(seed=seed)

    model = SimpleCNN(in_channels=3, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir = Path("checkpoints") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, history = 0.0, []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val["loss"], "val_acc": val["acc"],
                        "val_f1": val["macro_f1"]})
        print(f"Epoca {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {val['loss']:.3f} acc {val['acc']:.3f} f1 {val['macro_f1']:.3f}")

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save(model.state_dict(), out_dir / "best.pth")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Cel mai bun F1 pe validare: {best_f1:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--name", default="cnn")
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    main(epochs=args.epochs, lr=args.lr, name=args.name)