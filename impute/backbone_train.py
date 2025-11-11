import warnings
import re
from sklearn.preprocessing import LabelEncoder
import argparse
from guided_diffusion import logger
from guided_diffusion.datasets_loader import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    diffusion_defaults,
    create_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.cell_model import DiT_model
from guided_diffusion.train_util import TrainLoop
import pandas as pd
import torch
import numpy as np
import random
import os
import scanpy as sc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_argparser(index,steps,save_steps,batch):
        defaults = dict(
            data_dir=f"../impute/dataset/{index}/processed_data/process_data.h5ad",
            log_dir="./log_dir/diffusion/",
            log_suffix=f"_{index}",
            loss_path=f"./dataset/{index}/output/diffusion_model/loss_data.csv",
            model_name=f"train_model_DiT",
            save_dir=f'./dataset/{index}/output/diffusion_model/',
            lr_anneal_steps=steps,
            save_interval=save_steps,
            log_interval=save_steps//2,
            batch_size=batch,
            microbatch=-1,  # -1 disables microbatches
            lr=1e-4,
            weight_decay=0.0001,
            schedule_sampler="uniform",
            train_loss_dict={},
            ema_rate="0.9999",  # comma-separated list of EMA values
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
            ae_path=f"./dataset/{index}/output/muris_AE_model/best_model_dim=1024.pt",
            input_size=1024,
            hidden_dim=[2048],
        )
        defaults.update(diffusion_defaults())
        defaults['noise_schedule'] = "cosine"
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser

def main():
    setup_seed(1234)
    data_index = 'sim.groups10'
    batch = 32
    ori_data=sc.read_h5ad(f"../impute/dataset/{data_index}/processed_data/process_data.h5ad")
    cell_num=ori_data.to_df().shape[0]
    steps = cell_num // batch * 200 + 1
    save_steps = cell_num // batch * 100
    if 'celltype' in ori_data.obs.columns:
        celltype = ori_data.obs['celltype']
        label_encoder = LabelEncoder()
        label_encoder.fit(celltype)
        classes = label_encoder.transform(celltype)
        num_classes = celltype.nunique()
    else:
        classes = None
        num_classes = 1

    args = create_argparser(data_index,steps,save_steps,batch).parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir + args.model_name, exist_ok=True)
    logger.configure(dir=args.log_dir, log_suffix=args.log_suffix)
    logger.log("creating model and diffusion...")
    diffusion = create_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )
    model=DiT_model(
        input_size=args.input_size,
        depth=6,
        dit_type='dit',
        num_heads=8,
        classes=num_classes,
        mlp_ratio=2,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        label=classes,
        ae_path=args.ae_path,
        latent_dim = args.input_size,
        hidden_dim = args.hidden_dim,
        load_type="train",

    )
    logger.log("training...")
    train_loss_dict=TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        train_loss_dict=args.train_loss_dict,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        model_name=args.model_name,
        save_dir=args.save_dir,
    ).run_loop()

    train_df = pd.DataFrame(list(train_loss_dict.items()), columns=['Step', 'Train_Loss'])
    train_df.to_csv(args.loss_path, index=False)


if __name__ == "__main__":
        main()


