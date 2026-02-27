import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from cnn import CNN


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
    acc, y_counts, pred_counts, cm = evaluate(model, test_loader, device=device)

    print("\nTEST:")
    print("test y counts   :", y_counts)
    print("test pred counts:", pred_counts)
    print(f"test acc = {acc:.3f}")
    print("confusion matrix (rows=true, cols=pred):\n", cm)
    

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

    acc = correct / total if total > 0 else 0.0
    return acc, y_counts.numpy(), pred_counts.numpy(), cm.numpy()

if __name__ == "__main__":
    main()
