from __future__ import annotations

import argparse
import json
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CRPS-D slot detector on ParkRecon3D BEV prepared data")
    parser.add_argument("--train-dir", required=True, help="Directory with .jpg/.json generalized marking points")
    parser.add_argument("--val-dir", required=True, help="Directory with .jpg/.json generalized marking points")
    parser.add_argument("--external-repo-path", default="external/CRPS-D")
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--output-path", default="models/parking_slot/parkrecon3d_finetuned.pth")
    parser.add_argument("--metadata-path", help="Optional path for JSON training metadata")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--depth-factor", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    install_visdom_stub()
    add_external_repo(args.external_repo_path)

    import config as crpsd_config
    import data
    from model import TeacherDetector

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_dataset = data.ParkingSlotDatasetWithLabel(args.train_dir)
    val_dataset = data.ParkingSlotDatasetWithLabel(args.val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: list(zip(*batch)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: list(zip(*batch)),
        pin_memory=device.type == "cuda",
    )

    model = TeacherDetector(3, args.depth_factor, crpsd_config.NUM_FEATURE_MAP_CHANNEL).to(device)
    state_dict = torch.load(args.pretrained_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        set_encoder_trainable(model, trainable=epoch > args.freeze_encoder_epochs)
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "encoder_frozen": epoch <= args.freeze_encoder_epochs,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_state_dict(args.output_path, model)

    metadata_path = Path(args.metadata_path) if args.metadata_path else Path(args.output_path).with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "crpsd_teacher_detector",
                "pretrained_path": args.pretrained_path,
                "output_path": args.output_path,
                "train_dir": args.train_dir,
                "val_dir": args.val_dir,
                "best_val_loss": best_val_loss,
                "history": history,
                "args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Saved best state_dict to {args.output_path}")
    print(f"Saved metadata to {metadata_path}")


def run_epoch(model, loader, optimizer, device: torch.device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_images = 0

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, marking_points in progress:
        images = torch.stack(images).to(device)
        objective, weights = generate_objective(marking_points, images.shape[0], device)

        with torch.set_grad_enabled(train):
            prediction = model(images)
            loss = weighted_mse_loss(prediction, objective, weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_images += batch_size
        progress.set_postfix(loss=float(loss.item()))

    return total_loss / max(1, total_images)


def generate_objective(marking_points_batch, batch_size: int, device: torch.device):
    feature_map_size = 16
    channels = 9
    objective = torch.zeros(batch_size, channels, feature_map_size, feature_map_size, device=device)
    weights = torch.zeros_like(objective)
    weights[:, 0].fill_(1.0)

    for batch_idx, marking_points in enumerate(marking_points_batch):
        for marking_point in marking_points:
            col = clamp_index(math.floor(marking_point.x * feature_map_size), feature_map_size)
            row = clamp_index(math.floor(marking_point.y * feature_map_size), feature_map_size)
            objective[batch_idx, 0, row, col] = 1.0
            objective[batch_idx, 1, row, col] = marking_point.shape
            objective[batch_idx, 2, row, col] = marking_point.x * feature_map_size - col
            objective[batch_idx, 3, row, col] = marking_point.y * feature_map_size - row
            objective[batch_idx, 4, row, col] = (math.cos(marking_point.direction0) + 1.0) / 2.0
            objective[batch_idx, 5, row, col] = (math.sin(marking_point.direction0) + 1.0) / 2.0
            objective[batch_idx, 6, row, col] = (math.cos(marking_point.direction1) + 1.0) / 2.0
            objective[batch_idx, 7, row, col] = (math.sin(marking_point.direction1) + 1.0) / 2.0
            objective[batch_idx, 8, row, col] = marking_point.type

            weights[batch_idx, 1:9, row, col].fill_(2.0)
            weights[batch_idx, 4:8, row, col].fill_(6.0)

    return objective, weights


def weighted_mse_loss(prediction: torch.Tensor, objective: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((prediction - objective) ** 2 * weights).sum() / weights.sum().clamp_min(1.0)


def clamp_index(index: int, size: int) -> int:
    return max(0, min(size - 1, index))


def set_encoder_trainable(model, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def save_state_dict(output_path: str, model) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def add_external_repo(path: str) -> None:
    external_repo_path = Path(path).resolve()
    if not external_repo_path.exists():
        raise FileNotFoundError(f"External CRPS-D repo not found: {external_repo_path}")
    if str(external_repo_path) not in sys.path:
        sys.path.insert(0, str(external_repo_path))


def install_visdom_stub() -> None:
    if "visdom" in sys.modules:
        return
    visdom = types.ModuleType("visdom")

    class Visdom:
        def __init__(self, *args, **kwargs) -> None:
            pass

    visdom.Visdom = Visdom
    sys.modules["visdom"] = visdom


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
