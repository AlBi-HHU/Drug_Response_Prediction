import random
#import json
import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from functions import *

class Dataset():
    def __init__(self, x_cell, y):
        self.y = y
        self.imputation_mask = np.isnan(self.y)
        self.impute_by_col()
        self.x_cell = x_cell
        self.n = x_cell.shape[0]

    def __len__(self):
        return self.n
              
    def __getitem__(self, idx):
        x_cell = torch.tensor(self.x_cell[idx])
        y = torch.tensor(self.y[idx])
        imputation_mask = torch.tensor(self.imputation_mask[idx])
        return x_cell, y, imputation_mask
    
    def impute_by_col(self):
        y_means = np.nanmean(self.y, axis=0)
        idx = np.where(np.isnan(self.y))
        self.y[idx] = np.take(y_means, idx[1])

# this is from https://github.com/unnir/CancelOut/blob/master/pytorch_example.ipynb and slightly changed
class CancelOut(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(input_size, requires_grad=True))

    def forward(self, x_cell):
        return x_cell*torch.sigmoid(self.weights.float())

class Parametrized_Net(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size, output_size, use_cancelout, act_fct=nn.ReLU(), dropout=0.5):
        super().__init__()
        self.activation = act_fct
        self.layers = nn.ModuleList()
        self.cancelout = CancelOut(input_size) if use_cancelout != 'False' else None
        for i in range(num_layers):
            if i == 0:
                in_feats, out_feats = input_size, hidden_size
            elif i == num_layers-1:
                in_feats, out_feats = hidden_size, output_size
            else:
                in_feats, out_feats = hidden_size, hidden_size
            
            if i != num_layers-1:
                self.layers.append(nn.Sequential(nn.Linear(in_feats, out_feats), self.activation, nn.Dropout(dropout)))
            else:
                self.layers.append(nn.Linear(in_feats, out_feats))
        
    def forward(self, x_cell, mask=None):
        def hook(grad):
            grad[mask] = torch.zeros_like(grad[mask])
            return grad

        if self.cancelout:
            x_cell = self.cancelout(x_cell)
        
        for layer in self.layers:
            x_cell = layer(x_cell)
        
        out = x_cell #torch.sigmoid(x_cell) # if output needs to be between 0 and 1, use 'torch.sigmoid(x_cell)' instead of just 'x_cell'
        
        if x_cell.requires_grad:
            out.register_hook(hook)
        return out

    def get_cancelout_feature_weights(self):
        if self.cancelout is None:
            return None
        w = list(self.cancelout.parameters())[0].detach().cpu()
        w = torch.sigmoid(w)
        w = w.numpy()
        return w

def train(model, optimizer, criterion, device, train_loader):
    model.train()
    for idx, (xs, ys, mask) in enumerate(train_loader): #tqdm(train_loader)):
        xs = xs.to(device)
        ys = ys.to(device)
        mask = mask.to(device)
        outputs = model(xs, mask)
        loss = criterion(outputs, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def test(model, device, test_loader):
    model.eval()
    y_pred = torch.Tensor().to(device)
    with torch.no_grad():
        for idx, (xs, _, _) in enumerate(test_loader): #tqdm(test_loader)):
            xs = xs.to(device)
            outputs = model(xs)
            y_pred = torch.cat((y_pred, outputs))
    y_pred = y_pred.numpy()
    return y_pred

if __name__ == "__main__":
    resampling = snakemake.wildcards['resampling']
    inputs = snakemake.wildcards['inputs']
    solver = snakemake.wildcards['s']
    x_cell, _, y, feature_names = process_inputs_and_output(inputs, solver) # feature_names stores selected features
    y_val_split = np.load(snakemake.input[0])
    y_test_split = np.load(snakemake.input[1])
    y_train, y_val, y_test = train_val_test_split(y, y_val_split, y_test_split, resampling)

    if resampling == 'recordwise':
        x_train = x_cell.copy()
        x_val = x_cell.copy()
        x_test = x_cell.copy()
    # remove all-nan rows that occur due to subject-wise resampling
    elif resampling == 'subjectwise':
        x_train = x_cell.copy()
        x_train[y_val_split] = np.nan
        x_train[y_test_split] = np.nan
        mask = np.all(np.isnan(x_train), axis=1)
        x_train = x_train[~mask]
        x_val = x_cell[y_val_split]
        x_test = x_cell[y_test_split]
    else: # resampling == 'targetwise'
        raise Exception('Target-wise cross-validation is not possible for MMLP!')

    new_feature_names = feature_names.copy() # for keeping features in second-last iteration (last iteration has a worse loss)
    batch_size = 8
    num_layers = 3
    hidden_size = 2048
    act_fct = nn.LeakyReLU()
    dropout = 0.5
    learning_rate = 0.0001
    #momentum = 0.95
    weight_decay = 0
    criterion = nn.MSELoss()
    #num_epochs = 10
    threshold = 10
    epochs = 0
    limit = 300
    use_cancelout = 'feature_ranking_only' # three options: feature_selection, feature_ranking_only, False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    best_model = None
    min_min_rmse = float('inf')
    min_min_rmse_ccc = None
    while True:
        train_dataset = Dataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1) # after we iterate over all batches the data is shuffled
        val_dataset = Dataset(x_val, x_val) # y is not needed, so we just use x_val for an array of the same shape
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=1)
        input_size = x_train.shape[1]
        output_size = y.shape[1]
        model = Parametrized_Net(num_layers, hidden_size, input_size, output_size, use_cancelout, act_fct, dropout).to(device)
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        min_rmse = float('inf')
        min_rmse_ccc = None
        counter = 0
        while epochs < limit and counter < threshold:
            potential_best_model = train(model, optimizer, criterion, device, train_loader)
            epochs += 1
            y_pred = test(model, device, val_loader)
            if resampling == 'recordwise':
                val_row_indices = y_val_split // output_size
                val_col_indices = y_val_split % output_size
                y_pred = y_pred[val_row_indices, val_col_indices]
            elif resampling == 'subjectwise':
                y_val = y[y_val_split].flatten()
                mask = np.isnan(y_val)
                y_pred = y_pred.flatten()
                y_val = y_val[~mask]
                y_pred = y_pred[~mask]
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            ccc = concordance_correlation_coefficient(y_val, y_pred)
            print(epochs, resampling, rmse, ccc)
            if rmse < min_rmse:
                min_rmse = rmse
                min_rmse_ccc = ccc
                best_model = copy.deepcopy(potential_best_model)
                counter = 0
            else:
                counter += 1
        if min_rmse < min_min_rmse:
            feature_names = new_feature_names.copy()
            min_min_rmse = min_rmse
            min_min_rmse_ccc = min_rmse_ccc
            if use_cancelout == 'feature_selection':
                percentage = 0.05 # kick 5 % worst features # TODO: maybe very small numbers have more influence than 0.5? because if no sigmoid, then we have strong negative and strong positive values, which have more impact than 0?
                w = best_model.get_cancelout_feature_weights()
                w_flat = w.flatten()
                #w_mask = w >= np.median(w_flat) # only keep better half (or more) of features # TODO: maybe rather 5 % steps instead of 50 % steps
                w_mask = w >= np.sort(w_flat)[int(percentage*w_flat.shape[0])]
                x_train = x_train[:, w_mask]
                x_val = x_val[:, w_mask]
                x_test = x_test[:, w_mask]
                new_feature_names = list(np.array(new_feature_names)[w_mask])
                # weights are resetted automatically above when initializing Parametrized_Net
            else:
                break
    with open(snakemake.output[0], 'w') as f:
        f.write(str(min_min_rmse) + ', ' + str(min_min_rmse_ccc))

    test_dataset = Dataset(x_test, x_test) # y is not needed, so we just use x_test for an array of the same shape
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1)
    y_pred = test(best_model, device, test_loader)
    if resampling == 'recordwise':
        test_row_indices = y_test_split // output_size
        test_col_indices = y_test_split % output_size
        y_pred = y_pred[test_row_indices, test_col_indices]
    elif resampling == 'subjectwise':
        y_test = y[y_test_split].flatten()
        mask = np.isnan(y_test)
        y_pred = y_pred.flatten()
        y_test = y_test[~mask]
        y_pred = y_pred[~mask]
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    ccc = concordance_correlation_coefficient(y_test, y_pred)
    print('final scores:', np.nanmean(rmse), ccc)
    with open(snakemake.output[1], 'w') as f:
        f.write(str(rmse))
    with open(snakemake.output[2], 'w') as f:
        f.write(str(ccc))
    torch.save(best_model.state_dict(), snakemake.output[3])
    """features = {}
    features['num_selected_features'] = len(feature_names)
    features['selected_features'] = feature_names
    with open(snakemake.output[4], 'w') as f:
        json.dump(features, f)"""