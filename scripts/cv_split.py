import json
import numpy as np
import pandas as pd

if __name__ == "__main__":
    inputs = snakemake.wildcards['inputs']
    inputs_cell, inputs_drug, output = inputs.split('.')
    n_splits = snakemake.config['n_splits']
    resampling = snakemake.wildcards['resampling']

    with open(f'features/shared_cell_lines/{inputs_cell}.{output}_cell_lines.json', 'r') as f:
        cell_lines = json.load(f)
    y = pd.read_csv(f'targets/{output}.csv', index_col=0)
    y = y.loc[cell_lines]
    y = y.to_numpy()

    n, p = y.shape
    np.random.seed(0)

    if resampling == 'subjectwise':
        index_length = n
    elif resampling == 'recordwise':
        index_length = n*p
    elif resampling == 'targetwise':
        index_length = p

    indices = np.arange(index_length)
    if resampling == 'recordwise':
        mask = np.isnan(y.flatten())
        indices = indices[~mask]
        index_length -= np.sum(mask)

    np.random.shuffle(indices)
    for k in range(n_splits):
        test_indices_k = indices[(k*index_length)//n_splits:((k+1)*index_length)//n_splits]
        np.save(snakemake.output[k], test_indices_k)
        if k != n_splits-1:
            val_indices_k = indices[((k+1)*index_length)//n_splits:((k+2)*index_length)//n_splits]
        else:
            val_indices_k = indices[:index_length//n_splits]
        np.save(snakemake.output[n_splits+k], val_indices_k)