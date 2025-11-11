# scCADI: Cell-Aware Diffusion for Single-Cell RNA-seq Imputation
a conditional diffusion-based imputation framework for scRNA-seq data
## Introduction
scCADI, a conditional diffusion-based imputation framework for scRNA-seq data. The framework aims to achieve robust imputation across multiple datasets and application scenarios while facilitating downstream analyses. In scCADI, an autoencoder first projects each cell into the latent space, where the observed data and metadata are integrated as conditional priors to guide the diffusion process in learning biologically meaningful expression patterns. With this design, scCADI achieves accurate gene expression data recovery while preserving intrinsic biological variability among cells.
## Requirements
```text
Python==3.8.0   
torch==1.11.0
```
You can install dependencies with:
```bash
pip install -r requirements.txt
```
## Usage
We can quickly start scCADI:
### Step 1: process data
data process
```bash
python data_process.py
```
### Step 2: train model
XXX


