# scCADI: Cell-Aware Diffusion for Single-Cell RNA-seq Imputation
a conditional diffusion-based imputation framework for scRNA-seq data
## Introduction
scCADI, a conditional diffusion-based imputation framework for scRNA-seq data. At the training stage, an autoencoder first projects each cell into the latent space, where the observed data and metadata information as conditioning factors to guide the denoising process in learning biologically meaningful expression patterns. At the imputation stage, cell-wise latent embeddings are generated and decoded to impute the identified dropout sites, thereby recovering accurate gene expression profiles. With this design, scCADI achieves robust imputation across multiple datasets and application scenarios while facilitating downstream analyses.
## Requirements
```text
python==3.8.0   
pytorch==1.11.0
numpy==1.24.4
anndata==0.9.2
scanpy==1.9.8
scikit-learn==1.3.2
pandas==2.0.3
mpi4py==4.0.0
tqdm==4.66.5
```
You can install dependencies with:
```bash
pip install -r requirements.txt
```
## Usage
We can quickly start scCADI:
### Step 1: process data
We need a .h5ad file or .csv file of scRNA-seq dataset, where each row represents a cell and each column corresponds to a gene.The `impute/data_process.py` Python script is used to preprocess data, obtaining the pre-processed data used for training the model is formatted in h5ad.
```bash
python data_process.py
```
### Step 2: train model
XXX


