import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import create_dataloaders, prepare_dataset
from model import (
    build_model,
    build_model_uncertainty,
    cmixmatch,
    linear_rampup,
    mixmatch,
    nt_xent_loss,
    semi_supervised_loss,
)

try:
    import wandb
except ImportError:
    wandb = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Limited-data regression example with supervised, SSL, and semi-supervised modes.")
    parser.add_argument("--train_pct", type=float, default=5.0, help="Percentage of data used for labeled training.")
    parser.add_argument("--val_pct", type=float, default=10.0, help="Percentage of data used for validation.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dataset_source",
        type=str,
        default="auto",
        choices=["auto", "digits", "utkface"],
        help="Use UTK-Face when available, otherwise fall back to the built-in digits dataset.",
    )
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Enable Weights and Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="learning_with_less", help="Weights and Biases project name.")
    parser.add_argument(
        "--mode",
        type=str,
        default="supervised",
        choices=["supervised", "self_supervised", "semi_supervised"],
        help="Training mode.",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for contrastive loss.")
    parser.add_argument("--lambda_u", type=float, default=10.0, help="Weight for the unlabeled loss.")
    parser.add_argument(
        "--semi_supervised_mode",
        type=str,
        default="mixmatch",
        choices=["mixmatch", "cmixmatch"],
        help="Semi-supervised training algorithm.",
    )
    parser.add_argument(
        "--waitepochs",
        type=int,
        default=4,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--sequential_training",
        default=False,
        action="store_true",
        help="First run self-supervised pretraining, then fine-tune with the selected mode.",
    )
    parser.add_argument("--self_supervised_epochs", type=int, default=4, help="Epochs used during self-supervised pretraining.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for checkpoints and plots.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, criterion, optimizer, device, args, transform=None, unlabel_loader=None, global_step=0):
    model.train()
    running_loss = 0.0
    additional_metrics = {}
    num_batches = 0

    if args.mode == "supervised":
        for batch in tqdm(train_loader, desc="Training", leave=False):
            inputs = batch["image"].to(device)
            targets = batch["age"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

    elif args.mode == "self_supervised":
        for batch in tqdm(unlabel_loader, desc="Training", leave=False):
            views1 = [transform(img) for img in batch["image"]]
            views2 = [transform(img) for img in batch["image"]]

            view1 = torch.stack(views1).to(device)
            view2 = torch.stack(views2).to(device)

            optimizer.zero_grad()
            z1 = model(view1)
            z2 = model(view2)

            loss = nt_xent_loss(z1, z2, args.temperature)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

    elif args.mode == "semi_supervised":
        train_iter = iter(train_loader)
        for unlabel_batch in tqdm(unlabel_loader, desc="Training", leave=False):
            try:
                labeled_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                labeled_batch = next(train_iter)

            x_l, y_l = labeled_batch["original_image"], labeled_batch["age"].to(device)
            u = unlabel_batch["image"]
            current_lambda = args.lambda_u * linear_rampup(global_step)
            optimizer.zero_grad()

            if args.semi_supervised_mode == "mixmatch":
                (labeled_inputs, true_labels), (unlabeled_inputs, guessed_labels) = mixmatch(
                    labeled_batch=(x_l, y_l),
                    unlabeled_batch=u,
                    model=model,
                    augment_fn=transform,
                    T=args.temperature,
                    K=2,
                    alpha=0.75,
                    mode="regression",
                    device=device,
                )
            else:
                (labeled_inputs, true_labels), (unlabeled_inputs, guessed_labels) = cmixmatch(
                    labeled_batch=(x_l, y_l),
                    unlabeled_batch=u,
                    model=model,
                    augment_fn=transform,
                    T=args.temperature,
                    K=2,
                    alpha=0.75,
                    device=device,
                )

            output_l = model(labeled_inputs)
            output_u = model(unlabeled_inputs)
            loss, loss_x, loss_u = semi_supervised_loss(
                labeled_output=output_l,
                labeled_target=true_labels,
                unlabeled_output=output_u,
                unlabeled_target=guessed_labels,
                lambda_u=current_lambda,
                criterion=criterion,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            additional_metrics = {
                "labeled_loss": loss_x.item(),
                "unlabeled_loss": loss_u.item(),
                "current_lambda": current_lambda,
            }
            num_batches += 1
            global_step += 1

    epoch_loss = running_loss / max(num_batches, 1)
    return epoch_loss, additional_metrics, global_step


def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["image"].to(device)
            targets = batch["age"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / max(len(dataloader.dataset), 1)


def save_history(history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry.get("val_loss") for entry in history]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, marker="o", label="train")
    if any(loss is not None for loss in val_loss):
        plt.plot(epochs, val_loss, marker="s", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning With Less: training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_model(model, train_loader, val_loader, criterion, optimizer, device, args, transform=None, unlabel_loader=None, vis_val=True, tag=None):
    best_loss = float("inf")
    global_step = 0
    wait_epoch = args.waitepochs
    epochs_without_improvement = 0
    history = []
    tag = tag or args.mode

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        train_loss, additional_metrics, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            args=args,
            transform=transform,
            unlabel_loader=unlabel_loader,
            global_step=global_step,
        )

        if args.mode != "self_supervised":
            val_loss = eval_model(model, val_loader, criterion, device)
            current_loss = val_loss
        else:
            val_loss = None
            current_loss = train_loss

        if current_loss < best_loss:
            best_loss = current_loss
            checkpoint_path = args.results_dir / f"best_model_{tag}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "global_step": global_step,
        }
        metrics.update(additional_metrics)
        history.append(metrics)

        print(f"Epoch {epoch}/{args.epochs} -> Train Loss: {train_loss:.4f}", end="")
        if vis_val and val_loss is not None:
            print(f" | Val Loss: {val_loss:.4f}", end="")
        if args.mode == "semi_supervised" and additional_metrics:
            print(f" | Current λ: {additional_metrics['current_lambda']:.4f}", end="")
        print("")
        print(f"Best loss so far: {best_loss:.4f}")

        if args.use_wandb:
            wandb.log({k: v for k, v in metrics.items() if v is not None})

        if epochs_without_improvement >= wait_epoch:
            print(f"Early stopping triggered after {wait_epoch} epochs without improvement.")
            break

    history_path = args.results_dir / f"{tag}_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    save_history(history, args.results_dir / f"{tag}_loss.png")
    print(f"Training complete. Best loss: {best_loss:.4f}")
    return history


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or run without --use_wandb.")
        wandb.init(project=args.wandb_project, config=vars(args))

    ds_train, ds_val, ds_unlabel = prepare_dataset(
        args.train_pct,
        args.val_pct,
        dataset_source=args.dataset_source,
        seed=args.seed,
    )
    train_loader, unlabel_loader, val_loader, augment_transform = create_dataloaders(
        ds_train, ds_val, ds_unlabel, args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    if args.sequential_training:
        print("Starting sequential training: self-supervised pretraining followed by fine-tuning.")
        self_supervised_model = build_model(self_supervised=True).to(device)
        optimizer_ss = optim.Adam(self_supervised_model.parameters(), lr=args.lr)

        original_mode = args.mode
        original_epochs = args.epochs
        args.mode = "self_supervised"
        args.epochs = args.self_supervised_epochs

        train_model(
            self_supervised_model,
            train_loader,
            val_loader,
            criterion,
            optimizer_ss,
            device,
            args,
            augment_transform,
            unlabel_loader,
            vis_val=False,
            tag="self_supervised_pretrain",
        )

        pretrained_encoder = self_supervised_model.encoder
        args.mode = original_mode
        args.epochs = original_epochs

        downstream_model = build_model(model=pretrained_encoder, self_supervised=False).to(device)
        optimizer = optim.Adam(downstream_model.parameters(), lr=args.lr)
        train_model(
            downstream_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            args,
            augment_transform,
            unlabel_loader,
            tag=f"sequential_{args.mode}",
        )
    else:
        if args.mode == "self_supervised":
            model = build_model(self_supervised=True).to(device)
        else:
            model = build_model_uncertainty(self_supervised=False).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            args,
            augment_transform,
            unlabel_loader,
        )


if __name__ == "__main__":
    main()
