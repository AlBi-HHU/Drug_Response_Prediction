import random
#import json
import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool, GATConv, max_pool
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions import *
from TGDRPid import TGDRP, train, test

class Dataset():
    def __init__(self, x_cell, cell_indices, x_drug, drug_indices, y):
        self.y = y
        self.x_cell = x_cell
        self.cell_indices = cell_indices
        self.x_drug = x_drug
        self.drug_indices = drug_indices
        self.n = y.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x_cell = self.cell_indices[idx]
        x_drug = self.drug_indices[idx]
        y = self.y[idx]
        return x_cell, x_drug, y
  
class SA(nn.Module):
    def __init__(self, drug_nodes, cell_nodes, drug_edges, cell_edges, device):
        super(SA, self).__init__()
        self.drug_nodes_feature = drug_nodes.to(device)
        self.drug_edges = drug_edges.to(device)
        self.cell_nodes_feature = cell_nodes.to(device)
        self.cell_edges = cell_edges.to(device)
        self.dropout_ratio = 0.2
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.dim_cell = cell_nodes.size(1)
        self.dim_drug = drug_nodes.size(1)
        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio)
        )
        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio)
        )
        self.drug_conv = SAGEConv(self.dim_drug, 256)
        self.cell_conv_1 = SAGEConv(self.dim_cell, 1024)
        self.cell_conv_2 = SAGEConv(1024, 256)
        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(512, 1)
        )

    def forward(self, cell, drug):
        drug_id = drug.long()
        cell_id = cell.long()
        drug_x = self.drug_nodes_feature
        cell_x = self.cell_nodes_feature
        drug_x = self.drug_emb(drug_x)
        # drug_x = self.dropout(F.relu(self.drug_conv(drug_x, self.drug_edges)))
        cell_x = self.dropout(F.relu(self.cell_conv_1(cell_x, self.cell_edges)))
        cell_x = self.dropout(F.relu(self.cell_conv_2(cell_x, self.cell_edges)))
        # cell_x = self.cell_emb(cell_x)
        drug_x = drug_x.squeeze()
        cell_x = cell_x.squeeze()
        x = torch.cat([drug_x[drug_id], cell_x[cell_id]], -1)
        x = self.regression(x)
        return x

def load_graph_data_SA(x_cell, x_drug, y, inputs, path_to_TGDRP_model):
    drug_graph = x_drug
    cell_graph = x_cell
    genes_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'
    input_size_drug = x_drug[0].x.shape[1]
    _, _, output = inputs.split('.')
    if 'gdsc' in output:
        dataset = 'gdsc'
    else: # elif 'ccle' in output:
        dataset = 'ccle'
    cluster_predefine = get_predefine_cluster(x_cell[0].edge_index, 0.95, device, dataset)
    model = TGDRP(cluster_predefine, input_size_drug, len(inputs.split('.')[0].split('-'))).to(device)
    model.load_state_dict(torch.load(path_to_TGDRP_model))
    parameter = {'drug_emb': model.drug_emb, 'cell_emb': model.cell_emb, 'regression': model.regression}
    drug_nodes = model.GNN_drug(Batch.from_data_list(drug_graph).to(device)).detach()
    cell_nodes = model.GNN_cell(Batch.from_data_list(cell_graph).to(device)).detach()

    knn = 5 # default value is 5 in TGSA paper
    cell_edges, drug_edges = get_knn(x_cell, x_drug, y, knn, inputs)
    return drug_nodes, cell_nodes, drug_edges, cell_edges, parameter

if __name__ == "__main__":
    resampling = snakemake.wildcards['resampling']
    inputs = snakemake.wildcards['inputs']
    solver = snakemake.wildcards['s']
    split = snakemake.wildcards['k']
    n_splits = snakemake.config['n_splits']
    path_to_TGDRP_model = f'{n_splits}fold_cv/{inputs}/resampling_{resampling}/TGDRPid/TGDRPid_split{split}_best_model.pt'
    x_cell, x_drug, y, _ = process_inputs_and_output(inputs, solver)
    y_val_split = np.load(snakemake.input[0])
    y_test_split = np.load(snakemake.input[1])
    y_train, y_val, y_test = train_val_test_split(y, y_val_split, y_test_split, resampling)
    if resampling == 'recordwise':
        n, p = y.shape

        cell_train_indices_raw = np.repeat(np.arange(n), p, axis=0)
        drug_train_indices_raw = np.tile(np.arange(p), n)
        y_train_ext = y_train.flatten()
        y_train_ext = y_train_ext.reshape(-1, 1)
        # remove y nan rows in training data
        y_train_ext_mask = np.where(y_train_ext == y_train_ext)[0]
        cell_train_indices = cell_train_indices_raw[y_train_ext_mask]
        drug_train_indices = drug_train_indices_raw[y_train_ext_mask]
        y_train_ext = y_train_ext[y_train_ext_mask]

        cell_val_indices = cell_train_indices_raw[y_val_split]
        drug_val_indices = drug_train_indices_raw[y_val_split]
        y_val_ext = y_val.reshape(-1, 1)
        y_val_ext_mask = np.where(y_val_ext == y_val_ext)[0]
        cell_val_indices = cell_val_indices[y_val_ext_mask]
        drug_val_indices = drug_val_indices[y_val_ext_mask]
        y_val_ext = y_val_ext[y_val_ext_mask]
        
        cell_test_indices = cell_train_indices_raw[y_test_split]
        drug_test_indices = drug_train_indices_raw[y_test_split]
        y_test_ext = y_test.reshape(-1, 1)
        y_test_ext_mask = np.where(y_test_ext == y_test_ext)[0]
        cell_test_indices = cell_test_indices[y_test_ext_mask]
        drug_test_indices = drug_test_indices[y_test_ext_mask]
        y_test_ext = y_test_ext[y_test_ext_mask]
    elif resampling == 'subjectwise':
        n, p = y.shape
        
        y_train_split = np.setdiff1d(np.setdiff1d(np.arange(n), y_val_split), y_test_split)
        cell_train_indices = np.repeat(y_train_split, p, axis=0)
        drug_train_indices = np.tile(np.arange(p), len(y_train_split))
        y_train_ext = y_train.flatten()
        y_train_ext = y_train_ext.reshape(-1, 1)
        # remove y nan rows in training data
        y_train_ext_mask = np.where(y_train_ext == y_train_ext)[0]
        cell_train_indices = cell_train_indices[y_train_ext_mask]
        drug_train_indices = drug_train_indices[y_train_ext_mask]
        y_train_ext = y_train_ext[y_train_ext_mask]
        
        cell_val_indices = np.repeat(y_val_split, p, axis=0)
        drug_val_indices = np.tile(np.arange(p), len(y_val_split))
        y_val_ext = y_val.flatten()
        y_val_ext = y_val_ext.reshape(-1, 1)
        # remove y nan rows in valing data
        y_val_ext_mask = np.where(y_val_ext == y_val_ext)[0]
        y_val_ext_nan_mask = np.where(y_val_ext != y_val_ext)[0]
        cell_val_indices = cell_val_indices[y_val_ext_mask]
        drug_val_indices = drug_val_indices[y_val_ext_mask]
        y_val_ext = y_val_ext[y_val_ext_mask]

        cell_test_indices = np.repeat(y_test_split, p, axis=0)
        drug_test_indices = np.tile(np.arange(p), len(y_test_split))
        y_test_ext = y_test.flatten()
        y_test_ext = y_test_ext.reshape(-1, 1)
        # remove y nan rows in testing data
        y_test_ext_mask = np.where(y_test_ext == y_test_ext)[0]
        y_test_ext_nan_mask = np.where(y_test_ext != y_test_ext)[0]
        cell_test_indices = cell_test_indices[y_test_ext_mask]
        drug_test_indices = drug_test_indices[y_test_ext_mask]
        y_test_ext = y_test_ext[y_test_ext_mask]
    elif resampling == 'targetwise':
        n, p = y.shape
        
        y_train_split = np.setdiff1d(np.setdiff1d(np.arange(p), y_val_split), y_test_split)
        cell_train_indices = np.repeat(np.arange(n), len(y_train_split), axis=0) #np.tile(np.arange(n), len(y_train_split))
        drug_train_indices = np.tile(y_train_split, n) #np.repeat(y_train_split, n, axis=0)
        y_train_ext = y_train.flatten()
        y_train_ext = y_train_ext.reshape(-1, 1)
        # remove y nan rows in training data
        y_train_ext_mask = np.where(y_train_ext == y_train_ext)[0]
        drug_train_indices = drug_train_indices[y_train_ext_mask]
        cell_train_indices = cell_train_indices[y_train_ext_mask]
        y_train_ext = y_train_ext[y_train_ext_mask]
        
        cell_val_indices = np.repeat(np.arange(n), len(y_val_split), axis=0) #np.tile(np.arange(n), len(y_val_split))
        drug_val_indices = np.tile(y_val_split, n) #np.repeat(y_val_split, n, axis=0)
        y_val_ext = y_val.flatten()
        y_val_ext = y_val_ext.reshape(-1, 1)
        # remove y nan rows in valing data
        y_val_ext_mask = np.where(y_val_ext == y_val_ext)[0]
        y_val_ext_nan_mask = np.where(y_val_ext != y_val_ext)[0]
        cell_val_indices = cell_val_indices[y_val_ext_mask]
        drug_val_indices = drug_val_indices[y_val_ext_mask]
        y_val_ext = y_val_ext[y_val_ext_mask]

        cell_test_indices = np.repeat(np.arange(n), len(y_test_split), axis=0) #np.tile(np.arange(n), len(y_test_split))
        drug_test_indices = np.tile(y_test_split, n) #np.repeat(y_test_split, n, axis=0)
        y_test_ext = y_test.flatten()
        y_test_ext = y_test_ext.reshape(-1, 1)
        # remove y nan rows in testing data
        y_test_ext_mask = np.where(y_test_ext == y_test_ext)[0]
        y_test_ext_nan_mask = np.where(y_test_ext != y_test_ext)[0]
        cell_test_indices = cell_test_indices[y_test_ext_mask]
        drug_test_indices = drug_test_indices[y_test_ext_mask]
        y_test_ext = y_test_ext[y_test_ext_mask]

    batch_size = 512 # 512 is default for TGSA, not 128 as it is for TGDRP
    learning_rate = 0.0001
    weight_decay = 0
    criterion = nn.MSELoss()
    threshold = 10
    epochs = 0
    limit = 300
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = Dataset(x_cell, cell_train_indices, x_drug, drug_train_indices, y_train_ext)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # after we iterate over all batches the data is shuffled
    val_dataset = Dataset(x_cell, cell_val_indices, x_drug, drug_val_indices, y_val_ext)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    drug_nodes_data, cell_nodes_data, drug_edges, cell_edges, parameter = load_graph_data_SA(x_cell, x_drug, y, inputs, path_to_TGDRP_model)
    model = SA(drug_nodes_data, cell_nodes_data, drug_edges, cell_edges, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_model = None
    min_rmse = float('inf')
    min_rmse_ccc = None
    counter = 0
    while epochs < limit and counter < threshold:
        potential_best_model = train(model, optimizer, criterion, device, train_loader)
        epochs += 1
        y_pred = test(model, device, val_loader)
        rmse = mean_squared_error(y_val_ext, y_pred, squared=False)
        ccc = concordance_correlation_coefficient(y_val_ext[:, 0], y_pred[:, 0])
        print(epochs, resampling, rmse, ccc)
        if rmse < min_rmse:
            min_rmse = rmse
            min_rmse_ccc = ccc
            best_model = copy.deepcopy(potential_best_model)
            counter = 0
        else:
            counter += 1
    with open(snakemake.output[0], 'w') as f:
        f.write(str(min_rmse) + ', ' + str(min_rmse_ccc))

    test_dataset = Dataset(x_cell, cell_test_indices, x_drug, drug_test_indices, y_test_ext)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    y_pred = test(best_model, device, test_loader)
    rmse = mean_squared_error(y_test_ext, y_pred, squared=False)
    ccc = concordance_correlation_coefficient(y_test_ext[:, 0], y_pred[:, 0])
    print('final scores:', np.nanmean(rmse), ccc)
    with open(snakemake.output[1], 'w') as f:
        f.write(str(rmse))
    with open(snakemake.output[2], 'w') as f:
        f.write(str(ccc))
    torch.save(best_model.state_dict(), snakemake.output[3])