import random
#import json
import copy
import numpy as np

from functions import *

if __name__ == "__main__":
    resampling = snakemake.wildcards['resampling']
    inputs = snakemake.wildcards['inputs']
    solver = snakemake.wildcards['s']
    x_cell, _, y, feature_names = process_inputs_and_output(inputs, solver) # feature_names stores selected features
    #print(np.sum(y != y), y.shape[0]*y.shape[1], np.sum(y != y)/(y.shape[0]*y.shape[1])) # percentage of missing values in y
    y_val_split = np.load(snakemake.input[0])
    y_test_split = np.load(snakemake.input[1])
    y_train, y_val, y_test = train_val_test_split(y, y_val_split, y_test_split, resampling)

    with open(snakemake.output[0], 'w') as f:
        f.write(str(0) + ', ' + str(0))

    if resampling == 'recordwise':
        output_size = y.shape[1]
        y_train_val = y.copy()
        test_row_indices = y_test_split // output_size
        test_col_indices = y_test_split % output_size
        y_train_val[test_row_indices, test_col_indices] = np.nan
        y_train_val_mean_row = np.nanmean(y_train_val, axis=1)
        y_train_val_mean_col = np.nanmean(y_train_val, axis=0)
        y_pred = []
        for r, c in zip(test_row_indices, test_col_indices):
            y_pred.append((y_train_val_mean_row[r] + y_train_val_mean_col[c])/2) # mean of row and col mean
        y_test_ext = y_test.reshape(-1, 1)
        y_pred = np.array(y_pred)[:, np.newaxis]
        #y_pred = np.full(fill_value=y_train_val_mean, shape=y_test_ext.shape) # mean of all entries
    elif resampling == 'subjectwise':
        y_train_val_mean = np.nanmean(np.vstack((y_train, y_val)), axis=0)
        y_train_val_mean_mask = np.where(y_train_val_mean == y_train_val_mean)[0] # if there is a nan value due to an unlucky split (happens very rarely), we do not make predictions for that row
        y_train_val_mean = y_train_val_mean[y_train_val_mean_mask]
        y_test = y_test[:, y_train_val_mean_mask]
        y_train_val_mean_tile = np.tile(y_train_val_mean, (y_test.shape[0], 1))
        y_test_ext = y_test.reshape(-1, 1)
        y_test_ext_mask = np.where(y_test_ext == y_test_ext)[0]
        y_test_ext = y_test_ext[y_test_ext_mask]
        y_pred = y_train_val_mean_tile.reshape(-1, 1)[y_test_ext_mask]
    else:
        y_train_val_mean = np.nanmean(np.hstack((y_train, y_val)), axis=1)
        y_train_val_mean_mask = np.where(y_train_val_mean == y_train_val_mean)[0] # if there is a nan value due to an unlucky split (happens very rarely), we do not make predictions for that row
        y_train_val_mean = y_train_val_mean[y_train_val_mean_mask]
        y_test = y_test[y_train_val_mean_mask]
        y_train_val_mean_tile = np.tile(y_train_val_mean, (y_test.shape[1], 1)).T
        y_test_ext = y_test.reshape(-1, 1)
        y_test_ext_mask = np.where(y_test_ext == y_test_ext)[0]
        y_test_ext = y_test_ext[y_test_ext_mask]
        y_pred = y_train_val_mean_tile.reshape(-1, 1)[y_test_ext_mask]
    rmse = mean_squared_error(y_test_ext, y_pred, squared=False)
    ccc = concordance_correlation_coefficient(y_test_ext[:, 0], y_pred[:, 0])
    print('final scores:', np.nanmean(rmse), ccc)
    with open(snakemake.output[1], 'w') as f:
        f.write(str(rmse))
    with open(snakemake.output[2], 'w') as f:
        f.write(str(ccc))
    torch.save(None, snakemake.output[3])