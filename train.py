from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from cnn import CNN
import os
import sys
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

def main():
    # Assumption:
    #  X: [f_bands, w_sample, w_total]
    #  y: [w_total, 1]
    
    # NOT SURE WHICH TO PLACE SO PLACE ALL FOR NOW
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    dataPath = './datasets/npz/data_preprocessed.npz'
    savePath = './model'

    # Load data
    if not os.path.exists(dataPath):
        print(f'{dataPath} does not exist')
        sys.exit(1)

    data = np.load(dataPath)
    X: np.ndarray = data["X"]
    y: np.ndarray = data["y"]
    
    # SANITY CHECK
    # y = np.random.permutation(y)
    
    # Check save path exists before training
    if not os.path.exists(savePath):
        print(f'{savePath} does not exist')
        sys.exit(1)
        
    # Get loader for training and validation set
    train_loader, val_loader, weights = prepareLoader(X, y, batch_size=200)
    
    # Use gpu for training if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model (defined in separate class)
    model = CNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)

    for epoch in range(5):
        model.train()
        
        train_y_counts = torch.zeros(2, dtype=torch.long)
        train_pred_counts = torch.zeros(2, dtype=torch.long)
        val_y_counts = torch.zeros(2, dtype=torch.long)
        val_pred_counts = torch.zeros(2, dtype=torch.long)
    
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            # TRAINING PREDICTIONS
            pred = logits.argmax(dim=1)

            train_y_counts += torch.bincount(yb, minlength=2).cpu()
            train_pred_counts += torch.bincount(pred, minlength=2).cpu()


        # validation
        model.eval()
        correct = total = 0
        
        # No backpropagation for validation 
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                
                val_y_counts += torch.bincount(yb, minlength=2).cpu()
                val_pred_counts += torch.bincount(pred, minlength=2).cpu()
                # break
            
                correct += (pred == yb).sum().item()
                total += yb.numel()
                
        print(f"\nEpoch {epoch+1} TRAIN:")
        print("train y counts   :", train_y_counts.numpy())
        print("train pred counts:", train_pred_counts.numpy())
        
        print(f"\nEpoch {epoch+1} VAL:")
        print("val y counts   :", val_y_counts.numpy())
        print("val pred counts:", val_pred_counts.numpy())
        
        print(f"\n val acc = {correct/total:.3f}")
        print()  
    
    # Save the model for later use
    savePath = os.path.join(savePath, 'checkpoint.pth')
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "num_classes": 2,
            "epoch": epoch + 1,  # optional improvement
        },  
        savePath
    )
    
    print(f'Saved model data at {savePath}')


def prepareLoader(X: np.ndarray, y: np.ndarray, batch_size):
    _, _, total_window = X.shape
    
    # Pytorch expects input of (N, C, W, H)
    # X -> [w_total, f_bands, w_sample]
    # y -> [w_total, 1]
    X_torch = torch.tensor(X, dtype=torch.float32).permute(2, 0, 1).unsqueeze(1)  
    y_torch = torch.tensor(y, dtype=torch.long).view(-1)
    
    # print(type(X_torch))
    
    idx = np.arange(total_window) # [0,1,...,total_window-1]
    
    # Be aware that test set comes from the outside of the training set
    # (to handle cross-patient bias)
    train_idx, val_idx = train_test_split(
        idx, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True, 
        stratify=y_torch.numpy()
    )
    
    X_train, y_train = X_torch[train_idx], y_torch[train_idx]
    X_val,   y_val   = X_torch[val_idx],   y_torch[val_idx]
    
    # Compute weights to handle imbalance
    counts = torch.bincount(y_train)
    weights = (counts.sum() / counts).float()
    weights = weights / weights.sum() * len(counts)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, weights

if __name__ == "__main__":
    main()