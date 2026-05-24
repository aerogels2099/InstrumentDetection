import numpy as np
import torch
from sklearn.metrics import f1_score


def mixup_batch(x, y, alpha=0.4):
    """Linearly interpolate two random samples and their multi-label targets."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def train_one_epoch(model, loader, optimizer, criterion, device,
                    scheduler=None, scaler=None, mixup_alpha=0.4):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if mixup_alpha > 0:
            x, y = mixup_batch(x, y, alpha=mixup_alpha)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = criterion(out, y)
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step_taken = scaler.get_scale() >= scale_before
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_taken = True

        if scheduler is not None and step_taken:
            scheduler.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, threshold=0.5, class_thresholds=None):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(y.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    if class_thresholds is not None:
        preds = np.stack(
            [(all_probs[:, i] > class_thresholds[i]).astype(float)
             for i in range(all_probs.shape[1])],
            axis=1,
        )
    else:
        preds = (all_probs > threshold).astype(float)

    f1_per_class = [
        f1_score(all_labels[:, i], preds[:, i], zero_division=0)
        for i in range(all_labels.shape[1])
        if all_labels[:, i].sum() > 0
    ]

    return float(np.mean(f1_per_class)) if f1_per_class else 0.0


def find_optimal_thresholds(model, loader, device, num_classes):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(y.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    best_thresholds = []
    for i in range(num_classes):
        if all_labels[:, i].sum() == 0:
            best_thresholds.append(0.5)
            continue

        best_f1, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs[:, i] > thresh).astype(float)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        best_thresholds.append(best_thresh)

    return best_thresholds
