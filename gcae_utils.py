import numpy                    as np
import scipy.sparse             as sp
from scipy.linalg               import pinvh
from graph_embedding_config     import *
import torch
import scipy
import pickle


def load_data():
    # build graph
    temp = open(node_index_filename, 'rb')
    node_index = pickle.load(temp)
    temp.close()

    N = len(node_index)
    edges_unordered = np.genfromtxt(edgelist_filename, dtype=np.int32)
    edges = np.array(edges_unordered).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), ([node_index[x] for x in edges[:, 0]], [node_index[x] for x in edges[:, 1]])),
                        shape=(N, N),
                        dtype=np.float32)
    features = sp.csr_matrix(np.identity(N, dtype=np.float32))
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + adj.T

    adj = normalize(adj)
    # temp = adj.todense()
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_inv = adj
    # features = torch.FloatTensor(new_features)

    # Calculating the inverse, comment the following 4 lines if you don't want to use inverse
    # print("Calculating inverse")
    # adj_inv = sp.csr_matrix(pinvh(temp))
    # print("Calculation finished")
    # adj_inv = sparse_mx_to_torch_sparse_tensor(adj_inv)

    return features, adj, adj_inv


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)