import random
#import json
import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool, GATConv, max_pool
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions import *

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
        x_cell = self.x_cell[self.cell_indices[idx]]
        x_drug = self.x_drug[self.drug_indices[idx]]
        y = self.y[idx]
        return x_cell, x_drug, y

def collate_fn(samples):
    x_cell, x_drug, y = map(list, zip(*samples))
    x_cell = Batch.from_data_list(x_cell)
    x_drug = Batch.from_data_list(x_drug)
    y = torch.tensor(np.array(y))
    return x_cell, x_drug, y

class GNN_cell(torch.nn.Module):
    def __init__(self, num_feature, layer_cell, dim_cell, cluster_predefine):
        super().__init__()
        self.num_feature = num_feature
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.cluster_predefine = cluster_predefine
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        # self.activations = torch.nn.ModuleList()

        for i in range(self.layer_cell):
            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)
            else:
                conv = GATConv(self.num_feature, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # True or False
            # activation = nn.PReLU(self.dim_cell)

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)
            # self.activations.append(activation)

    def forward(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            num_node = int(cell.x.size(0) / cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)

        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)

        return node_representation
    
    def grad_cam(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            if i == 0:
                cell_node = cell.x
                cell_node.retain_grad()
            num_node = int(cell.x.size(0) / cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)

        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell) # 382 (438, davon nur die 382 unique genes)*8 (dim_cell)

        return cell_node, node_representation

class GNN_drug(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug, input_size_drug):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()

        for i in range(self.layer_drug):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(input_size_drug, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))

            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug):
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_list.append(x)
        node_representation = self.JK(x_drug_list)
        x_drug = global_max_pool(node_representation, batch)
        return x_drug

class TGDRP(nn.Module):
    def __init__(self, cluster_predefine, input_size_drug, num_feature):
        super().__init__()
        self.batch_size = 128
        self.layer_drug = 3
        self.dim_drug = 128
        self.num_feature = num_feature # usually 3 for CNV, EXP, MUT, but for example if EXP alone, then num_feature is 1
        self.layer_cell = 3
        self.dim_cell = 8
        self.dropout_ratio = 0.2

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug, input_size_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )

        # cell graph branch
        self.GNN_cell = GNN_cell(self.num_feature, self.layer_cell, self.dim_cell, cluster_predefine)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.GNN_cell.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )

        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 1)
        )

    def forward(self, cell, drug):
        # forward drug
        x_drug = self.GNN_drug(drug)
        x_drug = self.drug_emb(x_drug)
        
        # forward cell
        x_cell = self.GNN_cell(cell)
        x_cell = self.cell_emb(x_cell)

        # combine drug feature and cell line feature
        x = torch.cat([x_drug, x_cell], -1)
        x = self.regression(x)

        return x

def train(model, optimizer, criterion, device, train_loader):
    model.train()
    for idx, (xs_cell, xs_drug, ys) in enumerate(tqdm(train_loader, desc='Iteration')):
        xs_cell = xs_cell.to(device)
        xs_drug = xs_drug.to(device)
        ys = ys.to(device)
        outputs = model(xs_cell, xs_drug)
        loss = criterion(outputs, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def test(model, device, test_loader):
    model.eval()
    y_pred = torch.Tensor().to(device)
    with torch.no_grad():
        for idx, (xs_cell, xs_drug, _) in enumerate(tqdm(test_loader, desc='Iteration')):
            xs_cell = xs_cell.to(device)
            xs_drug = xs_drug.to(device)
            outputs = model(xs_cell, xs_drug)
            y_pred = torch.cat((y_pred, outputs))
    y_pred = y_pred.numpy()
    return y_pred

if __name__ == "__main__":
    resampling = snakemake.wildcards['resampling']
    inputs = snakemake.wildcards['inputs']
    _, _, output = inputs.split('.')
    if 'gdsc' in output:
        dataset = 'gdsc'
    else: # elif 'ccle' in output:
        dataset = 'ccle'
    solver = snakemake.wildcards['s']
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

    batch_size = 128
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4) # after we iterate over all batches the data is shuffled
    val_dataset = Dataset(x_cell, cell_val_indices, x_drug, drug_val_indices, y_val_ext)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    input_size_drug = x_drug[0].x.shape[1]
    cluster_predefine = get_predefine_cluster(x_cell[0].edge_index, 0.95, device, dataset)
    model = TGDRP(cluster_predefine, input_size_drug, len(inputs.split('.')[0].split('-'))).to(device)
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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    y_pred = test(best_model, device, test_loader)
    rmse = mean_squared_error(y_test_ext, y_pred, squared=False)
    ccc = concordance_correlation_coefficient(y_test_ext[:, 0], y_pred[:, 0])
    print('final scores:', np.nanmean(rmse), ccc)
    with open(snakemake.output[1], 'w') as f:
        f.write(str(rmse))
    with open(snakemake.output[2], 'w') as f:
        f.write(str(ccc))
    torch.save(best_model.state_dict(), snakemake.output[3])