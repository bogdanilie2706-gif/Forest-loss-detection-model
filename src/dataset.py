import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_dataloaders(root="data", batch_size=64, seed=42, num_workers=2, image_size=64):
    resize = [transforms.Resize(image_size)] if image_size != 64 else []
    train_tf = transforms.Compose(resize + [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose(resize + [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # restul funcției rămâne neschimbat
    # Același set de date, încărcat de două ori cu transformări diferite
    ds_train = datasets.EuroSAT(root, transform=train_tf, download=True)
    ds_eval = datasets.EuroSAT(root, transform=eval_tf, download=False)

    targets = ds_train.targets
    indices = list(range(len(targets)))

    # 70% antrenare / 15% validare / 15% test, păstrând proporția claselor
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=targets, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50,
        stratify=[targets[i] for i in temp_idx], random_state=seed
    )

    train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(ds_eval, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Subset(ds_eval, test_idx), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, ds_train.classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Clase:", classes)
    print("Batch-uri train/val/test:",
          len(train_loader), len(val_loader), len(test_loader))
    images, labels = next(iter(train_loader))
    print("Forma unui batch:", images.shape)   # [64, 3, 64, 64]
    print("Etichete:", labels[:10])