import os
import json
import itertools
import numpy as np
import pandas as pd
from dgllife.utils import *
from rdkit import Chem
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# adapted from TGDRP/TGSA
def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    8 features are canonical, 2 features are from OGB
    """
    featurizer_funcs = ConcatFeaturizer([atom_type_one_hot,
                                         atom_degree_one_hot,
                                         atom_implicit_valence_one_hot,
                                         atom_formal_charge,
                                         atom_num_radical_electrons,
                                         atom_hybridization_one_hot,
                                         atom_is_aromatic,
                                         atom_total_num_H_one_hot,
                                         atom_is_in_ring,
                                         atom_chirality_type_one_hot,
                                         ])
    atom_feature = featurizer_funcs(atom)
    return atom_feature

# adapted from TGDRP/TGSA
def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    featurizer_funcs = ConcatFeaturizer([bond_type_one_hot,
                                         # bond_is_conjugated,
                                         # bond_is_in_ring,
                                         # bond_stereo_one_hot,
                                         ])
    bond_feature = featurizer_funcs(bond)

    return bond_feature

# adapted from TGDRP/TGSA
def ensp_to_hugo_map():
    protein_info = pd.read_csv('data/TGSA_Data/9606.protein.info.v11.0.txt', delimiter='\t')
    ensp_map = dict(zip(protein_info['protein_external_id'], protein_info['preferred_name']))
    return ensp_map

# adapted from TGDRP/TGSA
def get_STRING_graph(threshold, gene_list):
    save_path = f'features/edge_index_PPI_{threshold}.npy'

    # rewritten so that not the whole 400 GB file is loaded at once
    if not os.path.exists(save_path):
        ensp_map = ensp_to_hugo_map()
        threshold1000 = threshold*1000
        edge_index = []
        with open('data/TGSA_Data/protein.links.detailed.v11.0.txt', 'r') as f: # get this 400 GB file from https://string-db.org/cgi/download.pl and then store it in data/TGSA_Data/
            next(f)
            for i, line in enumerate(f): # 6 183 296 833 lines
                split_line = line.split(' ')
                node0 = split_line[0]
                if not node0.startswith('9606'):
                    continue
                node1 = split_line[1]
                if not node1.startswith('9606'):
                    continue
                try:
                    combined_score = int(split_line[-1])
                except:
                    continue
                if combined_score > threshold1000:
                    if node0 in ensp_map.keys() and node1 in ensp_map.keys():
                        gene0 = ensp_map[node0]
                        gene1 = ensp_map[node1]
                        if gene0 in gene_list and gene1 in gene_list:
                            edge_index.append([gene_list.index(gene0), gene_list.index(gene1)])
                            edge_index.append([gene_list.index(gene1), gene_list.index(gene0)])
        edge_index = sorted(edge_index)
        edge_index = list(k for k,_ in itertools.groupby(edge_index))
        edge_index = np.array(edge_index, dtype=np.int64).T
        np.save(save_path, edge_index)
    else:
        edge_index = np.load(save_path)
    return edge_index

# adapted from TGDRP/TGSA
def get_predefine_cluster(edge_index, threshold, device, dataset):
    save_path = f'features/cluster_predefine_PPI_{threshold}.npy'
    if not os.path.exists(save_path):
        with open(f'features/cosmic_cancer_related_genes_{dataset}.json', 'r') as f:
            gene_list = json.load(f)
        g = Data(edge_index=torch.tensor(edge_index, dtype=torch.long), x=torch.zeros(len(gene_list), 1))
        g = Batch.from_data_list([g])
        cluster_predefine = {}
        for i in range(5):
            cluster = graclus(g.edge_index, None, g.x.size(0))
            g = max_pool(cluster, g, transform=None)
            cluster_predefine[i] = cluster
        np.save(save_path, cluster_predefine)
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}
    else:
        cluster_predefine = np.load(save_path, allow_pickle=True).item()
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}

    return cluster_predefine

# adapted from TGDRP/TGSA
def smiles2graph(smiles, solver):
    mol = Chem.MolFromSmiles(smiles)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 4  # value in original code was 3, but newer rdkit versions have 4 features for bond_type_one_hot
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # x_drug.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T
        # x_drug.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    else: # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = Data(x=torch.tensor(x, dtype=torch.float),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr), dtype=torch.float) # can be omitted, not used by GraphDRP, TGDRP, and TGSA

    return graph

def dummy_drug_graph(num_drugs, i):
    # create a graph for each drug i that has only one node which differs between all drugs (by setting a feature i to 1)
    x = np.zeros((1, num_drugs), dtype=np.int64)
    x[0, i] = 1
    edge_index = np.empty((2, 0), dtype=np.int64)
    edge_attr = np.empty((0, 0), dtype=np.int64)

    graph = Data(x=torch.tensor(x, dtype=torch.float),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr), dtype=torch.float) # can be omitted, not used by GraphDRP, TGDRP, and TGSA

    return graph

# adapted from TGSA
def get_knn(x_cell, x_drug, y, knn, inputs):
    # calculate pccs of gene expression, "draw" edges to top knn values
    inputs_cell, inputs_drug, output = inputs.split('.')
    inputs_cell_list = inputs_cell.split('-')
    with open(f'features/shared_cell_lines/{inputs_cell}.{output}_cell_lines.json', 'r') as f:
        cell_lines = json.load(f)
    
    y = pd.read_csv(f'targets/{output}.csv', index_col=0)
    y = y.loc[cell_lines]
    
    if 'gdsc' in output:
        dataset = 'gdsc'
    else: # elif 'ccle' in output:
        dataset = 'ccle'
    
    with open(f'features/cosmic_cancer_related_genes_{dataset}.json', 'r') as f:
        cosmic_cancer_related_genes = json.load(f)

    exp = pd.read_csv('features/cell_features/exp_tpm.csv', index_col=0, low_memory=False)
    cosmic_cancer_related_genes_j = [c + '.exp' for c in cosmic_cancer_related_genes]
    exp = exp.reindex(cell_lines) # needed in the case that not all cell lines in an entry of inputs_cell_list (for example MUT or CNV) appear in EXP
    exp = exp.loc[:, cosmic_cancer_related_genes_j]
    exp = exp.to_numpy(dtype=np.float32) # pytorch cannot handle float64/double or objects
    cell_edges_list = []
    exp_pccs = np.array(pd.DataFrame(exp).T.corr())
    for i in range(exp_pccs.shape[0]):
        top_indices = np.argpartition(exp_pccs[i], -knn)[-knn:]
        for j in top_indices:
            cell_edges_list.append([i, j])

    # calculate Jaccard similarity of Extended Connectivity Fingerprints (ECFP), "draw" edges to top knn values
    _, inputs_drug, _ = inputs.split('.')
    smiles = pd.read_csv(f'features/drug_smiles/{inputs_drug}.csv', index_col=1) # column 1 is PubchemID
    smiles = smiles.loc[y.columns.astype(int)]
    canonical_smiles = []
    for s in smiles.iloc[:, 1]: # [i, 0]: drug name, [i, 1]: canonical smiles, [i, 2]: isomeric smiles
        canonical_smiles.append(s)
    morgan_featurizer = dc.feat.CircularFingerprint() # is ECFP4
    morgan_features = morgan_featurizer.featurize(canonical_smiles)
    drug_edges_list = []
    jaccard_similarities = 1 - pairwise_distances(morgan_features.astype(bool), metric='jaccard')
    for i in range(jaccard_similarities.shape[0]):
        top_indices = np.argpartition(jaccard_similarities[i], -knn)[-knn:]
        for j in top_indices:
            drug_edges_list.append([i, j])
        drug_edges_list.append([i, j])
    
    drug_edges = torch.tensor(drug_edges_list, dtype=torch.long).t()
    cell_edges = torch.tensor(cell_edges_list, dtype=torch.long).t()
    
    return cell_edges, drug_edges

def process_inputs_and_output(inputs, solver):
    inputs_cell, inputs_drug, output = inputs.split('.')
    inputs_cell_list = inputs_cell.split('-')
    with open(f'features/shared_cell_lines/{inputs_cell}.{output}_cell_lines.json', 'r') as f:
        cell_lines = json.load(f)
    
    y = pd.read_csv(f'targets/{output}.csv', index_col=0)
    y = y.loc[cell_lines]
    
    if 'gdsc' in output:
        dataset = 'gdsc'
    else: # elif 'ccle' in output:
        dataset = 'ccle'
    
    x_cell = []
    x_drug = []
    feature_names = []
    if solver == 'MMLP' or solver == 'MMLPcosmic' or solver == 'MMLPid':
        if solver == 'MMLPcosmic': #the following code is for testing on COSMIC genes only (those that are used by TGDRP/TGSA and appear in the GDSC2 dataset)
            with open(f'features/cosmic_cancer_related_genes_{dataset}.json', 'r') as f:
                cosmic_cancer_related_genes = json.load(f)
            for j, i in enumerate(inputs_cell_list):
                x_cell_i = pd.read_csv(f'features/cell_features/{i}.csv', index_col=0, low_memory=False)
                if 'cnv' in i:
                    cosmic_cancer_related_genes_j = [c + '.cnv' for c in cosmic_cancer_related_genes]
                elif 'exp' in i:
                    cosmic_cancer_related_genes_j = [c + '.exp' for c in cosmic_cancer_related_genes]
                else: # if 'mut' in i:
                    cosmic_cancer_related_genes_j = [c + '.mut' for c in cosmic_cancer_related_genes]
                x_cell_i = x_cell_i.loc[cell_lines, cosmic_cancer_related_genes_j]
                feature_names.append(list(x_cell_i.columns))
                x_cell_i = x_cell_i.to_numpy(dtype=np.float32) # pytorch cannot handle float64/double or objects
                x_cell.append(x_cell_i)
            x_cell = np.concatenate(x_cell, axis=1)
        
        elif solver == 'MMLP' or solver == 'MMLPid':
            for i in inputs_cell_list:
                x_cell_i = pd.read_csv(f'features/cell_features/{i}.csv', index_col=0, low_memory=False)
                x_cell_i = x_cell_i.loc[cell_lines]
                feature_names.append(list(x_cell_i.columns))
                x_cell_i = x_cell_i.to_numpy(dtype=np.float32) # pytorch cannot handle float64/double or objects
                x_cell.append(x_cell_i)
            x_cell = np.concatenate(x_cell, axis=1)
        
        # scale x_cell columns between 0 and 1 (necessary/advisable for neural networks)
        if (x_cell < 0).any() or (x_cell > 1).any():
            x_cell_min = np.nanmin(x_cell, axis=0)
            x_cell_max = np.nanmax(x_cell, axis=0)
            x_cell_maxmin = x_cell_max - x_cell_min
            x_cell_maxmin[x_cell_maxmin == 0] = 1 # if all values in a column are the same, we divide by 0, so we leave those columns unchanged by dividing by 1
            x_cell = (x_cell - x_cell_min)/(x_cell_maxmin)
    if solver == 'TGDRP' or solver == 'TGSA' or solver == 'TGDRPid' or solver == 'TGSAid':
        threshold = 0.95
        with open(f'features/cosmic_cancer_related_genes_{dataset}.json', 'r') as f:
            cosmic_cancer_related_genes = json.load(f)
        edge_index = get_STRING_graph(threshold, cosmic_cancer_related_genes)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        cn_ex_mu = []
        scaler = StandardScaler()
        imp_mean = SimpleImputer()
        for i in inputs_cell_list:
            x_cell_i = pd.read_csv(f'features/cell_features/{i}.csv', index_col=0)
            x_cell_i.columns = [c[:-4] for c in x_cell_i.columns] # remove .mut, .cnv, .exp from column names for gene graph
            x_cell_i = x_cell_i.loc[cell_lines, cosmic_cancer_related_genes]
            feature_names.append(x_cell_i.columns)
            x_cell_i = x_cell_i.to_numpy()
            if 'cn' in i or 'ex' in i:
                x_cell_i = scaler.fit_transform(x_cell_i)
            if 'ex' in i:
                x_cell_i = imp_mean.fit_transform(x_cell_i) # only necessary if there are nan values in exp data (but we removed them beforehand in preprocess_data.ipynb)
            cn_ex_mu.append(x_cell_i)
        for i in range(len(cell_lines)):
            x_cell.append(Data(x=torch.tensor(np.vstack([cn_ex_mu[j][i] for j in range(len(inputs_cell_list))]), dtype=torch.float).T, edge_index=edge_index))
    if solver == 'TGDRP' or solver == 'TGSA' or solver == 'TGDRPid' or solver == 'TGSAid':
        smiles = pd.read_csv(f'features/drug_smiles/{inputs_drug}.csv', index_col=1) # column 1 is PubchemID
        smiles = smiles.loc[y.columns.astype(int)]
        x_drug = []
        for i, s in enumerate(smiles.iloc[:, 1]): # [i, 0]: drug name, [i, 1]: canonical smiles, [i, 2]: isomeric smiles
            if solver == 'TGDRPid' or solver == 'TGSAid': # baseline test 3: for testing TGDRP/TGSA with dummy molecule graph
                x_drug.append(dummy_drug_graph(len(smiles), i))
            else:
                x_drug.append(smiles2graph(s, solver))
    
    y = y.to_numpy(dtype=np.float32) # pytorch cannot handle float64/double or objects
    if len(feature_names) > 0: # if mean_predictor, feature_names is an empty list
        feature_names = list(np.concatenate(feature_names))
    
    if solver == 'MMLPid': # baseline test 2: for testing MMLP with dummy identity matrix
        x_cell = np.eye(x_cell.shape[0], dtype=np.float32)
        feature_names = [i for i in range(x_cell.shape[0])]
    return x_cell, x_drug, y, feature_names

def train_val_test_split(y, y_val_split, y_test_split, resampling):
    p = y.shape[1]
    if resampling == 'recordwise':
        val_row_indices = y_val_split // p
        val_col_indices = y_val_split % p
        test_row_indices = y_test_split // p
        test_col_indices = y_test_split % p
        y_train = y.copy()
        y_train[val_row_indices, val_col_indices] = np.nan
        y_train[test_row_indices, test_col_indices] = np.nan
        y_val = y[val_row_indices, val_col_indices]
        y_test = y[test_row_indices, test_col_indices]
    elif resampling == 'subjectwise':
        y_train = y.copy()
        y_train[y_val_split] = np.nan
        y_train[y_test_split] = np.nan
        y_val = y[y_val_split]
        y_test = y[y_test_split]
        mask = np.all(np.isnan(y_train), axis=1)
        y_train = y_train[~mask]
    else: # resampling == 'drugwise'
        y_train = y.copy()
        y_train[:, y_val_split] = np.nan
        y_train[:, y_test_split] = np.nan
        y_val = y[:, y_val_split]
        y_test = y[:, y_test_split]
        mask = np.all(np.isnan(y_train), axis=0)
        y_train = y_train[:, ~mask]
    return y_train, y_val, y_test

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    pearson_correlation_coefficient = pearsonr(y_true, y_pred)[0]
    ccc = (2*pearson_correlation_coefficient*std_true*std_pred)/(std_true**2 + std_pred**2 + (mean_true - mean_pred)**2)
    return ccc
