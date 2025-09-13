import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, features, labels):

        # convert to pytorch tensors
        self.features = torch.tensor(features, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)       
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label
