import torch
import torch.nn.functional as F
from torch import nn
from typing import List


class Encoder(nn.Module):
    """A class that encapsulates the encoder."""
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 256,
        hidden_dim: List[int] = [512],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        residual: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        n_genes: int
            The number of genes in the gene space, representing the input dimensions.
        latent_dim: int, default: 128
            The latent space dimensions
        hidden_dim: List[int], default: [1024, 1024]
            A list of hidden layer dimensions, describing the number of layers and their dimensions.
            Hidden layers are constructed in the order of the list for the encoder and in reverse
            for the decoder.
        dropout: float, default: 0.5
            The dropout rate for hidden layers
        input_dropout: float, default: 0.4
            The dropout rate for the input layer
        residual: bool, default: False
            Use residual connections.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual

        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(n_genes, hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.SiLU(),
                    )
                )
            else:
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.SiLU(),
                    )
                )
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x) -> F.Tensor:
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def save_state(self, filename: str):
        """Save state dictionary.

        Parameters
        ----------
        filename: str
            Filename to save the state dictionary.
        """
        torch.save({"state_dict": self.state_dict()}, filename)





class Decoder(nn.Module):
    """A class that encapsulates the decoder."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 256,
        hidden_dim: List[int] = [512],
        dropout: float = 0.5,
        residual: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        n_genes: int
            The number of genes in the gene space, representing the input dimensions.
        latent_dim: int, default: 128
            The latent space dimensions
        hidden_dim: List[int], default: [1024, 1024]
            A list of hidden layer dimensions, describing the number of layers and their dimensions.
            Hidden layers are constructed in the order of the list for the encoder and in reverse
            for the decoder.
        dropout: float, default: 0.5
            The dropout rate for hidden layers
        residual: bool, default: False
            Use residual connections.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.SiLU(),
                    )
                )
            else:
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.SiLU(),
                    )
                )
        self.network.append(nn.Linear(hidden_dim[-1], n_genes))

    def forward(self, x):
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

    def save_state(self, filename: str):
        """Save state dictionary.

        Parameters
        ----------
        filename: str
            Filename to save the state dictionary.
        """
        torch.save({"state_dict": self.state_dict()}, filename)


class AE(torch.nn.Module):
    def __init__(
        self,
        num_genes,
        device="cuda",
        latent_dim=128,
        hidden_dim=[1024,512],
        dropout=0.5,
        input_dropout=0.4,

    ):
        super(AE, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.device = device

        # set hyperparameters
        self.set_hparams_(latent_dim)

        # set models
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.residual = False
        self.encoder = Encoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            residual=self.residual,
        )
        self.decoder = Decoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=list(reversed(self.hidden_dim)),
            dropout=self.dropout,
            residual=self.residual,
        )

        # losses
        self.loss_autoencoder = nn.MSELoss(reduction='mean')

        self.iteration = 0

        self.to(self.device)

        # optimizers
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder, True)
            + get_params(self.decoder, True)
        )
        self.optimizer_autoencoder = torch.optim.AdamW(_parameters, lr=self.hparams["autoencoder_lr"], weight_decay=self.hparams["autoencoder_wd"],)


    def forward(self, data, return_latent=False, return_decoded=False):
        data = data.float()
        """
        If return_latent=True, act as encoder only. If return_decoded, genes should 
        be the latent representation and this act as decoder only.
        """
        if return_decoded:
            gene_reconstructions = self.decoder(data)
            gene_reconstructions = nn.ReLU()(gene_reconstructions)  # only relu when inference
            return gene_reconstructions

        latent_basal = self.encoder(data)
        if return_latent:
            return latent_basal

        gene_reconstructions = self.decoder(latent_basal)

        return gene_reconstructions



    def set_hparams_(self, latent_dim):

        self.hparams = {
            "dim": latent_dim,
            "autoencoder_width": 5000,
            "autoencoder_depth": 3,
            "adversary_lr": 3e-4,
            "autoencoder_wd": 0.01, 
            "autoencoder_lr": 5e-4, 
        }

        return self.hparams

    def masked_mean_flat(self,error_tensor, mask):
        masked_error = error_tensor * mask
        # Avoid division by zero
        denom = mask.sum(dim=list(range(1, mask.ndim)), keepdim=False).clamp(min=1.0)
        return masked_error.sum(dim=list(range(1, masked_error.ndim))) / denom
    def compute_loss(self, genes, genes_masked, obs_mask, cond_mask, alpha=0.5):

        genes = genes.to(self.device)
        genes_masked = genes_masked.to(self.device)
        obs_mask = obs_mask.to(self.device)
        cond_mask = cond_mask.to(self.device)

        gene_reconstructions = self.forward(genes_masked)

        target_mask = (obs_mask - cond_mask).to(self.device)
        masked_loss = self.masked_mean_flat((genes - gene_reconstructions) ** 2, target_mask).mean()
        reco_loss = self.loss_autoencoder(genes,gene_reconstructions)
        total_loss = alpha * masked_loss + (1 - alpha) * reco_loss

        return total_loss, masked_loss, reco_loss

    def train_AE(self, genes, genes_masked, obs_mask, cond_mask, alpha=0.5):

        total_loss, masked_loss, reco_loss = self.compute_loss(genes, genes_masked, obs_mask, cond_mask, alpha)

        self.optimizer_autoencoder.zero_grad()
        total_loss.backward()
        self.optimizer_autoencoder.step()

        self.iteration += 1

        return {
            "loss_total": total_loss.item(),
            "loss_masked": masked_loss.item(),
            "loss_reco": reco_loss.item(),
        }

    def eval_AE(self, genes, genes_masked, obs_mask, cond_mask, alpha=0.5):
        total_loss, masked_loss, reco_loss = self.compute_loss(genes, genes_masked, obs_mask, cond_mask, alpha)

        return {
            "loss_total": total_loss.item(),
            "loss_masked": masked_loss.item(),
            "loss_reco": reco_loss.item(),
        }
