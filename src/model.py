import torch
from torch import nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


if __name__ == "__main__":
    from dataset import get_dataloaders

    train_loader, _, _, classes = get_dataloaders(num_workers=0)
    model = SimpleCNN(in_channels=3, num_classes=len(classes))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri: {n_params:,}")

    images, labels = next(iter(train_loader))
    logits = model(images)
    print("Ieșire:", logits.shape)  # [64, 10]

    loss = nn.CrossEntropyLoss()(logits, labels)
    print(f"Loss inițial: {loss.item():.3f} (așteptat ~2.30)")