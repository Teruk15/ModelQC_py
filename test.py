import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from cnn import CNN

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc


def main():
    test_npz = "./datasets/npz/data_preprocessed.npz"
    ckpt_path = "./model/checkpoint.pth"

    if not os.path.exists(test_npz):
        raise FileNotFoundError(f"Missing test set: {test_npz}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN(num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loader = make_loader_from_npz(test_npz, batch_size=200)

    # UPDATED: evaluate now also returns y_true and y_score (prob of class 1)
    acc, y_counts, pred_counts, cm, y_true, y_score = evaluate(model, test_loader, device=device)

    print("\nTEST:")
    print("test y counts   :", y_counts)
    print("test pred counts:", pred_counts)
    print(f"test acc = {acc:.3f}")
    print("confusion matrix (rows=true, cols=pred):\n", cm)

    # NEW: save poster figures
    out_dir = "./figures"
    os.makedirs(out_dir, exist_ok=True)

    save_pretty_confusion_matrix(
        cm,
        out_path=os.path.join(out_dir, "confusion_matrix.png"),
        class_names=("Clean Channel", "Noisy Channel"),
        positive_class_index=1,  # Noisy is positive
    )
    save_roc_curve(y_true, y_score, out_path=os.path.join(out_dir, "roc_curve.png"))

    print(f"\nSaved figures to: {out_dir}/confusion_matrix.png and {out_dir}/roc_curve.png")


def make_loader_from_npz(npz_path: str, batch_size: int = 200):
    data = np.load(npz_path)
    X = data["X_test"]  # [f_bands, w_sample, w_total]
    y = data["y_test"]  # [w_total, 1]

    X_torch = torch.tensor(X, dtype=torch.float32).permute(2, 0, 1).unsqueeze(1)
    y_torch = torch.tensor(y, dtype=torch.long).view(-1)

    loader = DataLoader(TensorDataset(X_torch, y_torch), batch_size=batch_size, shuffle=False)
    return loader


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()

    correct = 0
    total = 0
    y_counts = torch.zeros(2, dtype=torch.long)
    pred_counts = torch.zeros(2, dtype=torch.long)

    # confusion matrix: rows=true, cols=pred
    cm = torch.zeros((2, 2), dtype=torch.long)

    # NEW: for ROC
    y_true_all = []
    y_score_all = []  # probability of class 1

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)

        correct += (pred == yb).sum().item()
        total += yb.numel()

        y_counts += torch.bincount(yb.cpu(), minlength=2)
        pred_counts += torch.bincount(pred.cpu(), minlength=2)

        for t, p in zip(yb.cpu(), pred.cpu()):
            cm[t, p] += 1

        # NEW: store scores for ROC (positive class = label 1)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(class==1)
        y_true_all.append(yb.detach().cpu().numpy())
        y_score_all.append(probs.detach().cpu().numpy())

    acc = correct / total if total > 0 else 0.0

    y_true_all = np.concatenate(y_true_all, axis=0) if len(y_true_all) else np.array([])
    y_score_all = np.concatenate(y_score_all, axis=0) if len(y_score_all) else np.array([])

    return acc, y_counts.numpy(), pred_counts.numpy(), cm.numpy(), y_true_all, y_score_all


def save_pretty_confusion_matrix(
    cm,
    out_path,
    class_names=("Clean Channel", "Noisy Channel"),
    positive_class_index=1,   # set 1 if "Noisy" is label 1
    figsize=(6.2, 5.6),       # a bit wider to reduce crowding
):
    """
    cm: numpy array shape (2,2), rows=true, cols=pred
    class_names: (name_for_label_0, name_for_label_1)
    positive_class_index: which label is considered "Positive" for TP/TN naming
        - If you want "Noisy" to be Positive, set positive_class_index to whatever index corresponds to "Noisy".
    """

    cm = np.asarray(cm)
    assert cm.shape == (2, 2), f"Expected (2,2) cm, got {cm.shape}"

    # Determine which index is negative/positive
    pos = int(positive_class_index)
    neg = 1 - pos

    neg_name, pos_name = class_names[neg], class_names[pos]

    # Helper to fetch counts given true/pred indices
    def C(t, p): return int(cm[t, p])

    # Positions in the grid (x=col, y=row)
    # row 0 is the top row visually after invert_yaxis(), so we draw y=0 for top.
    cells = []

    # True NEG row (true=neg)
    cells.append((0, 0, C(neg, neg), "True\nNegative",  True))   # pred=neg
    cells.append((1, 0, C(neg, pos), "False\nPositive", False))  # pred=pos

    # True POS row (true=pos)
    cells.append((0, 1, C(pos, neg), "False\nNegative", False))  # pred=neg
    cells.append((1, 1, C(pos, pos), "True\nPositive",  True))   # pred=pos

    fig, ax = plt.subplots(figsize=figsize)

    # Draw 2x2 grid manually
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    green = "#b6e3a8"
    red   = "#f4a6a6"

    for x, y, val, label, correct in cells:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=(green if correct else red)))
        # number (top-ish)
        ax.text(
            x + 0.5, y + 0.40, f"{val}",
            ha="center", va="center",
            fontsize=24, fontweight="bold"
        )
        # label (bottom-ish), two lines to save width
        ax.text(
            x + 0.5, y + 0.72, label,
            ha="center", va="center",
            fontsize=14, fontweight="bold"
        )

    # Axis ticks: map to actual class names in displayed order (neg on left/top, pos on right/bottom)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels([neg_name, pos_name], fontsize=14)

    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([neg_name, pos_name], fontsize=14)

    ax.set_xlabel("Predicted Label", fontsize=16)
    ax.set_ylabel("*True Label", fontsize=16)
    ax.set_title("Confusion Matrix", fontsize=20, pad=12)

    # Make it look clean
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, y_score, out_path: str):
    # If your test set ever has only one class, ROC is undefined—guard it.
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        print("WARNING: ROC curve not saved (need both classes present in y_true).")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()