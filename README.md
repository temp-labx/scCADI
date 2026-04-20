# scCADI: Cell-Aware Diffusion Model for Single-Cell RNA-seq Imputation
## Introduction
scCADI is a deep learning framework for imputing missing values in single-cell RNA-seq data. This repository provides the implementation of the model, along with scripts for data processing, model training, and imputation. The framework is designed to support expression recovery and related analyses in single-cell studies.
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
### Step 1: Data Preprocess
We need a .h5ad file or .csv file of scRNA-seq dataset, where each row represents a cell and each column corresponds to a gene. The `impute/data_process.py` Python script is used to preprocess data, obtaining the pre-processed data used for training the model is formatted in h5ad.
```bash
python data_process.py
```
### Step 2: Train the Autoencoder
The `AE/AE_train.py` Python script is used to train autoencoder model, resulting in a trained AE.
```bash
python AE_train.py
```
### Step 3: Train the Conditional Diffusion model
The `impute/backbone_train.py` Python script is used to train conditional diffusion model in the latent space learned by the AE, incorporating prior biological knowledge as conditional input during training, to obtain the trained diffusion model.
```bash
python backbone_train.py
```
### Step 4: Sample and Imputation
The `impute/impute.py` Python script is used to generate cell-wise latent embeddings from the trained diffusion model. These embeddings are decoded to recover dropout values and obtain the imputed scRNA-seq data.
```bash
python impute.py
```

