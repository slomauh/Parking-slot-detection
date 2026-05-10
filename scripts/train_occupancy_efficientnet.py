from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.occupancy.classifier import build_efficientnet_b0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 occupancy classifier")
    parser.add_argument("--data-dir", required=True, help="Directory with free/ and occupied/ subfolders")
    parser.add_argument("--val-dir", help="Optional validation directory with free/ and occupied/ subfolders")
    parser.add_argument("--output-path", default="models/occupancy/efficientnet_b0_crpsd.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
    if train_dataset.class_to_idx != {"free": 0, "occupied": 1}:
        raise ValueError(f"Expected class folders free/ occupied/, got {train_dataset.class_to_idx}")

    if args.val_dir:
        val_dataset = datasets.ImageFolder(args.val_dir, transform=eval_transform)
        if val_dataset.class_to_idx != train_dataset.class_to_idx:
            raise ValueError(
                f"Validation classes {val_dataset.class_to_idx} do not match train classes {train_dataset.class_to_idx}"
            )
    else:
        val_size = max(1, int(len(train_dataset) * args.val_ratio))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        eval_dataset = datasets.ImageFolder(args.data_dir, transform=eval_transform)
        train_dataset = train_subset
        val_subset.dataset = eval_dataset
        val_dataset = val_subset

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_efficientnet_b0(num_classes=2, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(json.dumps(row))

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.output_path, model, args, best_val_acc, history)

    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Saved best checkpoint to {args.output_path}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_size

    return total_loss / max(1, total), total_correct / max(1, total)


def save_checkpoint(output_path: str, model, args, best_val_acc: float, history: list[dict]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "efficientnet_b0",
            "classes": ["free", "occupied"],
            "best_val_acc": best_val_acc,
            "history": history,
            "args": vars(args),
        },
        path,
    )


if __name__ == "__main__":
    main()
