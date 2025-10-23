import torch
from torch.utils.data import Dataset, DataLoader
from config import BATCH_SIZE

class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float().unsqueeze(0)
        y = torch.tensor(self.y[idx])
        # ensure 3D shape: [1, num_features]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x, y

def make_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):
    train_loader = DataLoader(FlowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(FlowDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(FlowDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader
