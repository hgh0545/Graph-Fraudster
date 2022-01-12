from attack_model import BaseAttack
from torch.nn.modules.module import Module
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import utils

# class FGA(BaseAttack):
#
#     def __init__(self, model, nnodes):
#         super(FGA, self).__init__(model, nnodes)
#         self.nnodes= nnodes
#
#
#
#     def attack(self, adj, features, labels, idx_train, target_node, n_perturbations, device):
#         # self.init_adj(device)
#         self.adj_changes = Parameter(torch.FloatTensor(self.nnodes)).to(device)
#         self.adj_changes.data.fill_(0)
#         self.structure_perturbations = []
#         modified_adj = adj.todense()
#         features = utils.sparse_mx_to_torch_sparse_tensor(features)
#         modified_adj = torch.FloatTensor(modified_adj)
#         modified_adj = modified_adj.to(device=device)
#         features = features.to(device=device)
#         self.surrogate.eval()
#         # print(' number of pertubations: %s' % n_perturbations)
#
#
#         grad_mask = []
#         grad_mask.append(target_node)
#         for i in range(n_perturbations):
#             modified_row = modified_adj[target_node] + self.adj_changes
#             modified_adj[target_node] = modified_row
#             adj_norm = utils.normalize_adj_tensor(modified_adj)
#             output = self.surrogate(features, adj_norm)
#             loss = F.nll_loss(output[[target_node]], labels[[target_node]])
#             # loss = F.nll_loss(output[idx_train], labels[idx_train])
#             grad = torch.autograd.grad(loss, self.adj_changes, retain_graph=True)[0]
#             grad = grad * (-2 * modified_row + 1)
#             # grad[target_node] = 0
#             for mask in grad_mask:
#                 grad[mask] = -1
#             grad_argmax = torch.argmax(grad)
#             grad_mask.append(grad_argmax.item())
#             value = -2*modified_row[grad_argmax] + 1
#             modified_adj.data[target_node][grad_argmax] += value
#             modified_adj.data[grad_argmax][target_node] += value
#             self.structure_perturbations.append(tuple([target_node, grad_argmax.item()]))
#             # print([target_node, grad_argmax.item()])
#
#         modified_adj = modified_adj.detach().cpu().numpy()
#         modified_adj = sp.csr_matrix(modified_adj)
#         self.check_adj(modified_adj)
#         self.modified_adj = modified_adj
#
#     def init_adj(self, device):
#         self.adj_changes = Parameter(torch.FloatTensor(self.nnodes)).to(device)
#         self.adj_changes.data.fill_(0)

class FGA(BaseAttack):
    def __init__(self, model, nnodes):
        super(FGA, self).__init__(model, nnodes)
        self.nnodes= nnodes


    def attack(self, adj, features, labels, idx_train, target_node, n_perturbations, device):
        # self.init_adj(device)
        # self.adj_changes = Parameter(torch.FloatTensor(self.nnodes)).to(device)
        # self.adj_changes.data.fill_(0)
        self.structure_perturbations = []
        modified_adj = adj.todense()
        features = utils.sparse_mx_to_torch_sparse_tensor(features)
        modified_adj = torch.FloatTensor(modified_adj)
        modified_adj = modified_adj.to(device=device)
        features = features.to(device=device)
        self.surrogate.eval()
        # print(' number of pertubations: %s' % n_perturbations)

        modified_adj.requires_grad = True
        grad_mask = []
        grad_mask.append(target_node)
        for i in range(n_perturbations):
            # modified_row = modified_adj[target_node] + self.adj_changes
            # modified_adj[target_node] = modified_row
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = self.surrogate(features, adj_norm)
            loss = F.nll_loss(output[[target_node]], labels[[target_node]])
            # loss = F.nll_loss(output[idx_train], labels[idx_train])
            grad = torch.autograd.grad(loss, modified_adj)[0]
            grad = (grad[target_node] + grad[:, target_node]) * (-2 * modified_adj[target_node] + 1)
            grad[target_node] = -10
            grad_argmax = torch.argmax(grad)


            value = -2*modified_adj[target_node][grad_argmax] + 1
            modified_adj.data[target_node][grad_argmax] += value
            modified_adj.data[grad_argmax][target_node] += value
            self.structure_perturbations.append(tuple([target_node, grad_argmax.item()]))
            # print([target_node, grad_argmax.item()])

        modified_adj = modified_adj.detach().cpu().numpy()
        modified_adj = sp.csr_matrix(modified_adj)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj

    # def init_adj(self, device):
    #     self.adj_changes = Parameter(torch.FloatTensor(self.nnodes)).to(device)
    #     self.adj_changes.data.fill_(0)
class FGA1(BaseAttack):
    def __init__(self, model, nnodes):
        super(FGA1, self).__init__(model, nnodes)
        self.nnodes= nnodes


    def attack(self, adj, features, labels, idx_train, target_node, n_perturbations, device):
        # self.init_adj(device)
        # self.adj_changes = Parameter(torch.FloatTensor(self.nnodes)).to(device)
        # self.adj_changes.data.fill_(0)
        self.structure_perturbations = []
        modified_adj = adj.todense()
        features = utils.sparse_mx_to_torch_sparse_tensor(features)
        modified_adj = torch.FloatTensor(modified_adj)
        modified_adj = modified_adj.to(device=device)
        features = features.to(device=device)
        self.surrogate.eval()
        # print(' number of pertubations: %s' % n_perturbations)

        modified_adj.requires_grad = True
        grad_mask = []
        grad_mask.append(target_node)
        for i in range(n_perturbations):
            # modified_row = modified_adj[target_node] + self.adj_changes
            # modified_adj[target_node] = modified_row
            adj_norm1 = utils.normalize_adj_tensor_rgcn(modified_adj, -0.5)
            adj_norm2 = utils.normalize_adj_tensor_rgcn(modified_adj, -1)
            output = self.surrogate(features, adj_norm1, adj_norm2)
            loss = F.nll_loss(output[[target_node]], labels[[target_node]])
            # loss = F.nll_loss(output[idx_train], labels[idx_train])
            grad = torch.autograd.grad(loss, modified_adj)[0]
            grad = (grad[target_node] + grad[:, target_node]) * (-2 * modified_adj[target_node] + 1)
            grad[target_node] = -10
            grad_argmax = torch.argmax(grad)


            value = -2*modified_adj[target_node][grad_argmax] + 1
            modified_adj.data[target_node][grad_argmax] += value
            modified_adj.data[grad_argmax][target_node] += value
            self.structure_perturbations.append(tuple([target_node, grad_argmax.item()]))
            # print([target_node, grad_argmax.item()])

        modified_adj = modified_adj.detach().cpu().numpy()
        modified_adj = sp.csr_matrix(modified_adj)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj