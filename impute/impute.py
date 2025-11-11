import copy
import argparse
import torch.nn as nn
import warnings
import re
import anndata as ad
from sklearn.cluster import KMeans
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from guided_diffusion import logger
from guided_diffusion.script_util import (
    diffusion_defaults,
    create_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import scanpy as sc
import torch
import os
import pandas as pd
from guided_diffusion.datasets_loader import load_data
from guided_diffusion.cell_model import DiT_model
from AE.AE_model import AE
import random
def create_argparser(data_index):
    defaults = dict(
        clip_denoised=True,
        batch_size=32,
        use_ddim=False,
        log_dir="../impute/log_dir/sample/",
        log_suffix=f"_sample_{data_index}",
        model_path=f"./dataset/{data_index}/output/diffusion_model/train_model_DiT/model006200.pt",
        ori_data_path=f"./dataset/{data_index}/processed_data/process_data.h5ad",
        impute_data_path=f"./dataset/{data_index}/output/result/",
        ae_path=f"./dataset/{data_index}/output/muris_AE_model/best_model_dim=1024.pt",
        n_clusters=6,
        input_size=1024,
        hidden_dim=[2048],
    )
    defaults.update(diffusion_defaults())
    defaults['noise_schedule'] = "cosine"
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def replace_zeros_with_nan(series):
    if (series != 0).any():
        series = series.replace(0, np.nan)
    return series

def indentify(adata,n_clusters):
    data = adata.to_df()
    row_names = data.index
    col_names = data.columns
    if 'celltype' in adata.obs.columns:
        label =adata.obs['celltype'].to_frame(name='celltype')
        data.insert(loc=len(data.columns),
                    column='celltype',
                    value=label.values.flatten()
                    )
        data = data.groupby('celltype', group_keys=False).apply(
            lambda x: x.apply(replace_zeros_with_nan))
        data = data.drop(data.columns[-1], axis=1)
    else:
        sc.pp.pca(adata, n_comps=50)
        data_X = adata.obsm['X_pca']
        kmeans = KMeans(n_clusters=n_clusters).fit(data_X)
        adata.obs['cluster'] = list(kmeans.labels_)
        label = adata.obs['cluster'].to_frame(name='celltype')
        data.insert(loc=len(data.columns),
                    column='celltype',
                    value=label.values.flatten()
                    )
        data = data.groupby('celltype', group_keys=False).apply(
            lambda x: x.apply(replace_zeros_with_nan))
        data = data.drop(data.columns[-1], axis=1)
    observed_values = data.values.astype("float32")
    true_masks = np.isnan(observed_values)
    true_masks = np.where(true_masks, 1, 0)
    return true_masks,row_names,col_names


def main():
    setup_seed(1234)
    data_index = 'sim.groups10'
    ori_data_path= f"./dataset/{data_index}/processed_data/process_data.h5ad"
    ori_data=sc.read_h5ad(ori_data_path)
    impute_data_path = f"./dataset/{data_index}/output/result/"
    os.makedirs(impute_data_path, exist_ok=True)
    if 'celltype' in ori_data.obs.columns:
        celltype = ori_data.obs['celltype']
        label_encoder = LabelEncoder()
        label_encoder.fit(celltype)
        classes = label_encoder.transform(celltype)
        num_classes = celltype.nunique()
    else:
        classes = None
        num_classes = 1

    args = create_argparser(data_index).parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logger.configure(dir=args.log_dir, log_suffix=args.log_suffix)
    logger.log("creating model and diffusion...")
    diffusion = create_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )
    model = DiT_model(
        input_size=args.input_size,
        depth=6,
        dit_type='dit',
        num_heads=8,
        classes=num_classes,
        mlp_ratio=2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    autoencoder = AE(
        num_genes=ori_data.to_df().shape[1],
        device=device,
        latent_dim=args.input_size,
        hidden_dim=args.hidden_dim,
    )
    autoencoder.load_state_dict(torch.load(args.ae_path))
    autoencoder.to(device)
    autoencoder.eval()

    def model_fn(x, t, obs,label=None, **kwargs):
        return model(x, t, obs,label=label, **kwargs)

    """indetify dropout"""
    true_masks, row_names, col_names = indentify(ori_data, args.n_clusters)
    M_identify = pd.DataFrame(true_masks, index=row_names, columns=col_names).values

    logger.log("Load ori_data... ")
    ori_data_loader = load_data(
        data_dir=args.ori_data_path,
        batch_size=args.batch_size,
        label=classes,
        ae_path=args.ae_path,
        latent_dim =args.input_size,
        hidden_dim = args.hidden_dim,
        load_type="impute",
    )

    logger.log("sampling...")
    all_samples = []
    for i, (batch, labels) in enumerate(ori_data_loader):
        batch = batch.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        model_kwargs=labels
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model_fn,
            (batch.shape[0], batch.shape[1]),
            obs=batch,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            noise=None,
        )
        all_samples.append(sample.cpu().numpy())
    samples_arr = np.concatenate(all_samples, axis=0)
    samples_reco = autoencoder(torch.tensor(samples_arr).cuda(), return_latent=False, return_decoded=True)
    samples_reco = samples_reco.cpu().detach().numpy()
    reco_df = pd.DataFrame(samples_reco, index=ori_data.to_df().index, columns=ori_data.to_df().columns)
    reco_df = reco_df.clip(lower=0)
    """impute"""
    impute_df = ori_data.to_df().where(M_identify == 0, reco_df)
    impute_ad = ad.AnnData(X=impute_df)
    impute_ad.obs = ori_data.obs.copy()
    impute_ad.var = ori_data.var.copy()
    impute_ad.write_h5ad(impute_data_path+f'imputed_data.h5ad')
    logger.log("impute complete")



if __name__ == "__main__":
        main()

