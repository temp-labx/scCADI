import numpy as np
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
import torch
from AE.AE_model import AE

def load_AE(ae_path, num_gene, latent_dim,hidden_dim):
    autoencoder = AE(
        num_genes=num_gene,
        device='cuda',
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
    autoencoder.load_state_dict(torch.load(ae_path))
    return autoencoder
def load_data(
        *,
        data_dir,
        batch_size,
        label,
        ae_path,
        latent_dim,
        hidden_dim,
        load_type="train",

):

    if not data_dir:
        raise ValueError("unspecified cell data directory")
    import scipy.sparse

    # Load cell data
    adata = sc.read_h5ad(data_dir)

    if scipy.sparse.issparse(adata.X):
        cell_data = adata.X.toarray()
    else:
        cell_data = adata.X


    num_gene = cell_data.shape[1]
    autoencoder = load_AE(ae_path, num_gene, latent_dim,hidden_dim)
    autoencoder.eval()
    with torch.no_grad():
        cell_data = autoencoder(torch.tensor(cell_data).cuda(), return_latent=True)
        cell_data = cell_data.cpu().detach().numpy()


    dataset = scRNADataset(
        cell_data=cell_data,
        class_name=label,
    )

    if load_type!="impute":
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        # Infinite generator
        while True:
            yield from loader

    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # impute data is not shuffled
            num_workers=0,
            drop_last=False
        )
        # Generator
        yield from loader


class scRNADataset(Dataset):
    def __init__(self, cell_data,class_name):
        super().__init__()
        self.cell_data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.cell_data.shape[0]

    def __getitem__(self, idx):

        cell = self.cell_data[idx]
        out_dict = {}
        if self.class_name is not None:
            out_dict["label"] = np.array(self.class_name[idx], dtype=np.int64)
        return cell,out_dict
