import scanpy as sc
from scipy import sparse
import pandas as pd
import os
def extract_y_column(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        if 'x' in df.columns:
            y_list = df['x'].tolist()
        else:
            df = pd.read_csv(csv_file_path, header=None, names=['index', 'x'])
            y_list = df['x'].tolist()
    except pd.errors.ParserError:
        df = pd.read_csv(csv_file_path, header=None, names=['index', 'x'])
        y_list = df['x'].tolist()

    return y_list


def data_process_sim(data_path,label_path):
    adata=sc.read_csv(data_path).T
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    lable_elements = extract_y_column(label_path)
    adata.obs['celltype']=lable_elements
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    sc.pp.filter_genes(adata, min_cells = 1)
    sc.pp.filter_cells(adata, min_genes = 1)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata

def data_process_real(data_path):
    adata=sc.read_h5ad(data_path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    sc.pp.filter_genes(adata, min_cells = 1)
    sc.pp.filter_cells(adata, min_genes = 1)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    index = adata.var['highly_variable'].values
    adata = adata[:, index].copy()

    return adata

def main():
    """data_process for simulated dataset"""
    data_index = 'sim.groups10'
    data_path = f'./dataset/{data_index}/{data_index}_counts.csv'
    label_path = f'./dataset/{data_index}/{data_index}_groups.csv'
    file_path = f"./dataset/{data_index}/processed_data/"
    processed_data = data_process_sim(data_path, label_path)
    os.makedirs(file_path , exist_ok=True)
    processed_data.write_h5ad(file_path + f"process_data.h5ad")

    """data_process for real dataset"""
    # data_index = 'MP'
    # data_path = f'./dataset/{data_index}/raw_data.h5ad'
    # file_path = f"./dataset/{data_index}/processed_data/"
    # processed_data = data_process_real(data_path)
    # os.makedirs(file_path , exist_ok=True)
    # processed_data.write_h5ad(file_path + f"process_data.h5ad")


if __name__ == "__main__":
    main()



