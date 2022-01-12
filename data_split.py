import scipy.sparse as sp
import numpy as np
import torch

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # splits are random

    # Remove diagonal elements
    # print('adj', adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # print('adj', adj, adj.shape)
    # Check the diagonal elements are zeros:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # Sparse matrix with DIAgonal storage
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    # print('edges', edges, edges.shape)
    # print('edges_all', edges_all, edges_all.shape)

    # Link index
    num_val = int(edges.shape[0]*0.1)
    num_test = int(edges.shape[0]*0.6)
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    # print('test_edges', test_edges, test_edges.shape)
    # print('val_edges', val_edges, val_edges.shape)
    # print('train_edges', train_edges, train_edges.shape)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:,None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue  # Break this cycle
        if ismember([idx_i, idx_j], np.array(edges_all)):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], np.array(edges_all)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)  #~is Negate
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(train_edges, val_edges)
    assert ~ismember(val_edges, test_edges)
    assert ~ismember(train_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_trian = adj_train + adj_train.T

    # Note: these edge lists only contain sigle direction of edge!
    return adj_trian, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def split_features(args, feature):
    # num_feature_per_party = int(feature.shape[1]/args.num_parties)
    feature = np.array(feature.todense())
    if feature.shape[1] % args.num_parties != 0:
        del_v = np.random.randint(0,feature.shape[1],1*feature.shape[1] % args.num_parties)
        del_v = list(del_v)
        feature = np.delete(feature, del_v, axis=1)
        # expend_arr = np.zeros((feature.shape[0], int(feature.shape[1]//args.num_parties-feature.shape[1]%args.num_parties)))
        # feature = np.hstack([feature, expend_arr])

    # np.random.sample(range(0, feature.shape[1]), num_feature_per_party)
    featureT = feature.T
    np.random.shuffle(featureT)
    feat = featureT.T
    feature_list = np.split(feat, args.num_parties, axis=1)
    feature_list = [sp.csr_matrix(feat) for feat in feature_list]
    return feature_list

def split_graph(args, adj, feature, split_method, with_s=True, with_f=False):
    # Create new graph graph_A, graph_B
    # Function to build test set with 10% positive links
    # splits are random
    degrees = np.array(adj.todense()).sum(0)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # print('adj', adj, adj.shape)
    # Check the diagonal elements are zeros:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # Sparse matrix with DIAgonal storage
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]

    # print('edges', edges, edges.shape)
    # print('edges_all', edges_all, edges_all.shape)

    # Link index
    if with_s==True:
        if split_method == 'com':
            num_graph_A = int(edges.shape[0] * args.p)
            all_edge_idx = list(range(edges.shape[0]))
            np.random.shuffle(all_edge_idx)
            A_edge_idx = all_edge_idx[:num_graph_A]

            A_edges = edges[A_edge_idx]
            B_edges = np.delete(edges, A_edge_idx, axis=0)

        elif split_method == 'alone':
            num_graph_A = int(edges.shape[0] * args.p)
            num_graph_B = int(edges.shape[0] * args.q)
            all_edge_idx = list(range(edges.shape[0]))

            np.random.shuffle(all_edge_idx)
            A_edge_idx = all_edge_idx[:num_graph_A]
            np.random.shuffle(all_edge_idx)
            B_edge_idx = all_edge_idx[:num_graph_B]

            A_edges = edges[A_edge_idx]
            B_edges = edges[B_edge_idx]


        elif split_method == 'abs':
            A_edge_idx = []
            num_graph_A = int(edges.shape[0] * args.p)
            all_edge_idx = list(range(edges.shape[0]))
            np.random.shuffle(all_edge_idx)
            for i in all_edge_idx:
                if(degrees[edges[i][0]]>=2 and degrees[edges[i][1]]>=2):
                    A_edge_idx.append(i)
                    degrees[edges[i][0]]-=1
                    degrees[edges[i][1]]-=1
                if len(A_edge_idx) == num_graph_A:
                    break

            A_edges = edges[A_edge_idx]
            B_edges = np.delete(edges, A_edge_idx, axis=0)

        print(len(degrees.nonzero()[0]))


        data_A = np.ones(A_edges.shape[0])
        data_B = np.ones(B_edges.shape[0])

        # Re-build adj matrix
        adj_A = sp.csr_matrix((data_A, (A_edges[:, 0], A_edges[:, 1])), shape=adj.shape)
        adj_A = adj_A + adj_A.T
        degree_A = np.array(adj_A.sum(0))
        print(len(degree_A.nonzero()[0]))
        adj_B = sp.csr_matrix((data_B, (B_edges[:, 0], B_edges[:, 1])), shape=adj.shape)
        adj_B = adj_B + adj_B.T
        degree_B = np.array(adj_B.sum(0))
        print(len(degree_B.nonzero()[0]))
    else:
        adj_A = sp.csr_matrix(np.eye(adj.shape[0]))
        adj_B = sp.csr_matrix(np.eye(adj.shape[0]))
    # Feature split evenly
    # feature_A = torch.split(feature, feature.size()[1] // 2, dim=1)[0]
    # feature_B = torch.split(feature, feature.size()[1] // 2, dim=1)[1]
    if with_f==True:
        X_NUM = int(feature.shape[1] // 2)
        feature = np.array(feature.todense())
        feature_A = feature[:, :X_NUM]
        feature_B = feature[:,X_NUM:2*X_NUM]
    else:
        feature_A = np.eye(adj.shape[0])
        feature_B = np.eye(adj.shape[0])
    # feature_A = feature
    # feature_B = feature

    # adj_B_tui = adj_B
    # adj_A = preprocess_adj(adj_A)
    # adj_A = sparse_mx_to_torch_sparse_tensor(adj_A)

    # adj_B = preprocess_adj(adj_B)
    # adj_B = sparse_mx_to_torch_sparse_tensor(adj_B)
    if args.datasets=='polblogs':
        feature_A = np.eye(adj.shape[0])
        feature_B = np.eye(adj.shape[0])

    return adj_A, adj_B,  sp.csr_matrix(feature_A), sp.csr_matrix(feature_B)

def split_graph1(args, adj, feature, split_method):
    # Create new graph graph_A, graph_B
    # Function to build test set with 10% positive links
    # splits are random

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # print('adj', adj, adj.shape)
    # Check the diagonal elements are zeros:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # Sparse matrix with DIAgonal storage
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]

    # print('edges', edges, edges.shape)
    # print('edges_all', edges_all, edges_all.shape)

    # Link index
    if split_method == 'com':
        num_graph_A = int(edges.shape[0] * args.p)
        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)
        A_edge_idx = all_edge_idx[:num_graph_A]

        A_edges = edges[A_edge_idx]
        B_edges = np.delete(edges, A_edge_idx, axis=0)

    elif split_method == 'alone':
        num_graph_A = int(edges.shape[0] * args.p)
        num_graph_B = int(edges.shape[0] * args.q)
        all_edge_idx = list(range(edges.shape[0]))

        np.random.shuffle(all_edge_idx)
        A_edge_idx = all_edge_idx[:num_graph_A]
        np.random.shuffle(all_edge_idx)
        B_edge_idx = all_edge_idx[:num_graph_B]

        A_edges = edges[A_edge_idx]
        B_edges = edges[B_edge_idx]


    data_A = np.ones(A_edges.shape[0])
    data_B = np.ones(B_edges.shape[0])

    # Re-build adj matrix
    adj_A = sp.csr_matrix((data_A, (A_edges[:, 0], A_edges[:, 1])), shape=adj.shape)
    adj_A = adj_A + adj_A.T

    adj_B = sp.csr_matrix((data_B, (B_edges[:, 0], B_edges[:, 1])), shape=adj.shape)
    adj_B = adj_B + adj_B.T

    # Feature split evenly

    # feature_A = torch.split(feature, feature.size()[1] // 2, dim=1)[0]
    # feature_B = torch.split(feature, feature.size()[1] // 2, dim=1)[1]

    X_NUM = int(feature.shape[1] // 2)
    feature = np.array(feature.todense())
    feature_A = feature[:, :X_NUM]
    feature_B = feature[:,X_NUM:2*X_NUM]
    # feature_A = feature
    # feature_B = feature

    # adj_B_tui = adj_B
    # adj_A = preprocess_adj(adj_A)
    # adj_A = sparse_mx_to_torch_sparse_tensor(adj_A)

    # adj_B = preprocess_adj(adj_B)
    # adj_B = sparse_mx_to_torch_sparse_tensor(adj_B)

    return adj_A, adj_B,  sp.csr_matrix(feature_A), sp.csr_matrix(feature_B)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    # print('adj22',adj,type(adj))
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


def accuracy(pred, targ):
    pred = torch.max(pred, 1)[1]
    ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    return ac


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


