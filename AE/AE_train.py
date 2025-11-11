import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from AE_model import AE
import sys
sys.path.append("..")
from datasets_loader import load_train_valid
import scanpy as sc
torch.autograd.set_detect_anomaly(True)
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_vae(args):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_datasets, val_datasets = load_train_valid(
        data_dir=args["data_dir"],
        batch_size=args["batch_size"],
    )

    autoencoder = AE(
        num_genes=args["num_genes"],
        device=device,
        latent_dim=args['latent_dim'],
        hidden_dim=args['hidden_dim'],
    )
    return autoencoder, train_datasets, val_datasets,device

def get_randmask(observed_mask, mask_rate):
    cond_mask = torch.zeros_like(observed_mask)
    for i in range(len(observed_mask)):
        obs_idx = torch.where(observed_mask[i] > 0)[0]
        num_masked = round(len(obs_idx) * mask_rate)
        mask_idx = obs_idx[torch.randperm(len(obs_idx))[:num_masked]]
        cond_mask[i, obs_idx] = 1
        cond_mask[i, mask_idx] = 0
    return cond_mask




def train(args, return_model=False, patience=30):

    autoencoder, train_datasets, val_datasets,device = prepare_vae(args)
    args["hparams"] = autoencoder.hparams
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(args["save_dir"], f"best_model_dim={args['latent_dim']}.pt")

    train_losses, val_losses = [], []
    os.makedirs(args["save_dir"], exist_ok=True)

    for step in range(args["max_steps"]):
        autoencoder.train()
        train_loss = 0
        for genes, obs_mask in train_datasets:
            cond_mask = get_randmask(obs_mask, mask_rate=0.5)
            genes_masked = genes * cond_mask
            stats = autoencoder.train_AE(genes, genes_masked, obs_mask, cond_mask)
            train_loss += stats["loss_total"]

        avg_train_loss = train_loss / len(train_datasets)
        train_losses.append(avg_train_loss)

        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for genes, obs_mask in val_datasets:
                cond_mask = get_randmask(obs_mask, mask_rate=0.5)
                genes_masked = genes * cond_mask
                stats = autoencoder.eval_AE(genes, genes_masked, obs_mask, cond_mask)
                val_loss += stats["loss_total"]
        avg_val_loss = val_loss / len(val_datasets)
        val_losses.append(avg_val_loss)

        print(f"Step [{step}/{args['max_steps']}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(autoencoder.state_dict(), best_model_path)
            print("Validation loss improved, saving best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break



    if return_model:
        return autoencoder, train_datasets, val_datasets


def parse_arguments(num_genes,data_dir,save_dir):
    parser = argparse.ArgumentParser(description="train phase")
    parser.add_argument("--data_dir", type=str, default=data_dir)
    parser.add_argument("--num_genes", type=int, default=num_genes)
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=list, default=[2048])
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_minutes", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default=save_dir)
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    seed_everything(1234)
    data_index='sim.groups10'
    data_dir=f'../impute/dataset/{data_index}/processed_data/process_data.h5ad'
    num_genes = sc.read_h5ad(data_dir).to_df().shape[1]
    save_dir = f'../impute/dataset/{data_index}/output/muris_AE_model'
    train(parse_arguments(num_genes, data_dir, save_dir))



