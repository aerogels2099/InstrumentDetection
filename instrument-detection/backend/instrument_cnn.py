import os

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import InstrumentDataset
from model import InstrumentCNN14
from loss import AsymmetricLoss
from training import train_one_epoch, evaluate, find_optimal_thresholds

NUM_WORKERS = 4
PRETRAINED_PATH = os.path.join(os.path.dirname(__file__), "Cnn14_mAP=0.431.pth")

if __name__ == "__main__":
    train_csv = "data/train.csv"
    val_csv   = "data/val.csv"
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nLoading datasets...")
    train_dataset = InstrumentDataset(train_csv, augment=True)
    val_dataset   = InstrumentDataset(val_csv,   augment=False)

    loader_kwargs = dict(
        batch_size=32,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    df = pd.read_csv(train_csv)
    label_cols  = [c for c in df.columns if c not in ("file_path", "duration")]
    num_classes = len(label_cols)
    print(f"Number of classes:   {num_classes}")
    print(f"Training samples:    {len(train_dataset)}")
    print(f"Validation samples:  {len(val_dataset)}")

    pretrained = PRETRAINED_PATH if os.path.exists(PRETRAINED_PATH) else None
    if pretrained is None:
        print("Warning: PANNs checkpoint not found — training from scratch.")
        print("Run download_panns.py to fetch it.")

    model = InstrumentCNN14(num_classes=num_classes, pretrained_path=pretrained).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters:          {total:,} total  |  {trainable:,} trainable")

    criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, margin=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=100,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_f1          = 0.0
    patience         = 25
    patience_counter = 0
    f1_history       = []

    print("\nStarting training...")
    for epoch in range(100):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, scaler=scaler, mixup_alpha=0.4,
        )
        f1 = evaluate(model, val_loader, device)
        f1_history.append(f1)

        print(
            f"Epoch {epoch+1:>3}/100 | "
            f"Loss: {train_loss:.4f} | "
            f"Val F1: {f1:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Use rolling max over last 5 epochs to smooth noisy F1
        smoothed_f1 = max(f1_history[-5:])

        if smoothed_f1 > best_f1:
            best_f1 = smoothed_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1": f1,
            }, "best_instrument_cnn14.pth")
            print(f"           -> New best  F1: {smoothed_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs.")
            break

    print("\nFinding optimal thresholds...")
    checkpoint = torch.load("best_instrument_cnn14.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimal_thresholds = find_optimal_thresholds(model, val_loader, device, num_classes)
    print(f"Thresholds: {[f'{t:.2f}' for t in optimal_thresholds]}")

    final_f1 = evaluate(model, val_loader, device, class_thresholds=optimal_thresholds)
    print(f"Val F1 (calibrated): {final_f1:.4f}  |  Val F1 (0.5): {best_f1:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "thresholds": optimal_thresholds,
        "f1": final_f1,
        "num_classes": num_classes,
        "class_names": label_cols,
    }, "instrument_cnn14_final.pth")

    print("\nTraining complete. Model saved as 'instrument_cnn14_final.pth'")
