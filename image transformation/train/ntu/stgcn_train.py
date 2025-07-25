import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ====================
# Dataset
# ====================
class NTUDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)  # (N, C, T, V, M)
        with open(label_path, 'rb') as f:
            self.label_info = pickle.load(f)
        self.labels = self.label_info['label']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)  # (C, T, V, M)
        x = x.squeeze(-1)  # remove person dim -> (C, T, V)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ====================
# Simple ST-GCN Block
# ====================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, A.shape[0]))
        self.relu = nn.ReLU()
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride,1), padding=(4,0))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, A):
        # x: (N, C, T, V)
        # Apply Graph Conv (simple version: treat joints as channels)
        x = self.gcn(x)
        x = self.relu(x)
        x = self.tcn(x)
        return self.bn(x)

# ====================
# Simple ST-GCN Model
# ====================
class STGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25):
        super().__init__()
        # simple adjacency (identity)
        self.A = torch.eye(num_point)
        self.block1 = STGCNBlock(3, 64, self.A)
        self.block2 = STGCNBlock(64, 128, self.A, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_class)

    def forward(self, x):
        # x: (N, C, T, V)
        x = self.block1(x, self.A)
        x = self.block2(x, self.A)
        x = self.pool(x)  # -> (N, 128, 1, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ====================
# Training Script
# ====================
def train():
    data_path = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\STGCN_data\train_data.npy"
    label_path = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\STGCN_data\train_label.pkl"

    dataset = NTUDataset(data_path, label_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = STGCN(num_class=len(set(dataset.labels))).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # just 5 epochs for quick test
        total_loss, correct = 0, 0
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}: loss={total_loss:.4f}, acc={acc:.4f}")

if __name__ == "__main__":
    train()
