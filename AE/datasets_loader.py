import numpy as np
from torch.utils.data import DataLoader, Dataset,Subset
import scanpy as sc
import scipy.sparse


def load_train_valid(
        *,
        data_dir,
        batch_size,
):


    adata = sc.read_h5ad(data_dir)
    if scipy.sparse.issparse(adata.X):
        cell_data = adata.X.toarray()
    else:
        cell_data = adata.X
    cell_data = cell_data.astype('float32')
    obs_mask = (cell_data > 0).astype(np.float32)

    dataset = CellDataset(
        cell_data=cell_data,
        obs_mask=obs_mask,
    )

    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    split_point = int(num_samples * 0.8)
    train_indices = indices[:split_point]
    valid_indices = indices[split_point:]

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    return train_loader, valid_loader

class CellDataset(Dataset):
    def __init__(self, cell_data,obs_mask):

        super().__init__()
        self.cell_data = cell_data
        self.obs_mask = obs_mask


    def __len__(self):
        return self.cell_data.shape[0]

    def __getitem__(self, idx):

        cell = self.cell_data[idx]
        obs_mask=self.obs_mask[idx]
        return cell, obs_mask
