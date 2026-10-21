import torch
from torch import nn
from sklearn.metrics import f1_score


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


if __name__ == "__main__":
    # Rulezi asta O SINGURĂ DATĂ, la final, pe setul de test
    from sklearn.metrics import classification_report
    from dataset import get_dataloaders
    from model import SimpleCNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader, classes = get_dataloaders()

    model = SimpleCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load("checkpoints/best.pth", map_location=device))

    res = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Test: loss={res['loss']:.3f} acc={res['acc']:.3f} macro_f1={res['macro_f1']:.3f}")
    print(classification_report(res["labels"].numpy(), res["preds"].numpy(),
                                target_names=classes, digits=3))