import networkx as nx
import numpy as np
import os
import pickle
import torch
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

def process_features(features):
    row_sum_diag = np.sum(features, axis=1)
    row_sum_diag_inv = np.power(row_sum_diag, -1)
    row_sum_diag_inv[np.isinf(row_sum_diag_inv)] = 0.
    row_sum_inv = np.diag(row_sum_diag_inv)
    return np.dot(row_sum_inv, features)


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data1(dataset):
    ## get data
    if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
        data_path = 'data'
        suffixs = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph']
        objects = []
        for suffix in suffixs:
            file = os.path.join(data_path, 'ind.%s.%s'%(dataset, suffix))
            objects.append(pickle.load(open(file, 'rb'), encoding='latin1'))
        x, y, allx, ally, tx, ty, graph = objects
        x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()

        # test indices
        test_index_file = os.path.join(data_path, 'ind.%s.test.index'%dataset)
        with open(test_index_file, 'r') as f:
            lines = f.readlines()
        indices = [int(line.strip()) for line in lines]
        min_index, max_index = min(indices), max(indices)

        # preprocess test indices and combine all data
        tx_extend = np.zeros((max_index - min_index + 1, tx.shape[1]))
        features = np.vstack([allx, tx_extend])
        features[indices] = tx
        ty_extend = np.zeros((max_index - min_index + 1, ty.shape[1]))
        labels = np.vstack([ally, ty_extend])
        labels[indices] = ty
        labels1 = []
        for i in range(len(labels)):
            labels1.append(labels[i].argmax())
        labels1 = np.array(labels1)
        # get adjacency matrix
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()
        # adj = torch.from_numpy(adj)
        adj = np.array(adj)
        idx_train = np.arange(0, len(y), 1)
        idx_val = np.arange(len(y), len(y) + 500, 1)
        idx_test = np.array(indices)

        
    elif dataset == 'polblogs':
        adj = np.zeros((1222, 1222))
        with open('data/'+str(dataset) + '.txt')as f:
            for j in f:
                entry = [float(x) for x in j.split(" ")]
                adj[int(entry[0]), int(entry[1])] = 1
                adj[int(entry[1]), int(entry[0])] = 1
        labels1 = np.loadtxt('data/'+str(dataset) + '_label.txt')
        labels1 = labels1.astype(int)
        labels1 = labels1[:,1:].flatten()
        idx_train = np.loadtxt('data/'+str(dataset) + '_train_node.txt')
        idx_train = idx_train.astype(int)
        idx_val = np.loadtxt('data/'+str(dataset) + '_validation_node.txt')
        idx_val = idx_val.astype(int)
        idx_test = np.loadtxt('data/'+str(dataset) + '_test_node.txt')
        idx_test = idx_test.astype(int)

        features = np.eye(adj.shape[0])

    elif dataset == 'cora_ml':
        filename = 'data/' + str(dataset) + '_adj' + '.npz'
        adj = sp.load_npz(filename)
        filename = 'data/' + str(dataset) + '_features' + '.npz'
        features = sp.load_npz(filename)
        filename = 'data/' + str(dataset) + '_label' + '.npy'
        labels1 = np.load(filename)
        filename = 'data/' + str(dataset) + '_train_node' + '.npy'
        idx_train = np.load(filename)
        filename = 'data/' + str(dataset) + '_val_node' + '.npy'
        idx_val = np.load(filename)
        filename = 'data/' + str(dataset) + '_test_node' + '.npy'
        idx_test = np.load(filename)

    else:

        filename = 'data/' + 'amazon_electronics_photo' + '_adj' + '.npz'
        adj = sp.load_npz(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_features' + '.npz'
        features = sp.load_npz(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_label' + '.npy'
        labels1 = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo'+ '_train_node' + '.npy'
        idx_train = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_val_node' + '.npy'
        idx_val = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_test_node' + '.npy'
        idx_test = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_label' + '.npy'
        labels1 = np.load(filename)

    return sp.csr_matrix(adj), sp.csr_matrix(features), idx_train, idx_val, idx_test, labels1


def get_adj( filename, require_lcc=True):
    adj, features, labels = load_npz(filename)
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1

    if require_lcc:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels = labels[lcc]
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    # whether to set diag=0?
    adj.setdiag(0)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()

    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

    return adj, features, labels

def load_npz(file_name, is_sparse=True):
    with np.load(file_name) as loader:
        # loader = dict(loader)
        if is_sparse:
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                        loader['adj_indptr']), shape=loader['adj_shape'])
            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                             loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None
            labels = loader.get('labels')
        else:
            adj = loader['adj_data']
            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None
            labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)
    return adj, features, labels

def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):

    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test

def largest_connected_components(adj, n_components=1):
    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep