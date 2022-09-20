import argparse
import os
import torch
from gcn import GCN, Generator, Discriminator, SGC
import os
import os.path as osp
import torch.functional as F
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from datasets import load_data1
from utils import *
from torch.autograd import Variable
import time
from itertools import chain
# from attack_model import FGA
from FGA import FGA
from GF_Attack import GFA
from scipy import linalg
from nettack import Nettack
from sklearn.cluster import KMeans,MeanShift
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from gae import dot_product_decode
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_score,roc_curve,auc,average_precision_score,log_loss
import sklearn
import operator
from functools import reduce
from tqdm import tqdm
import similarity
import pylab
import seaborn as sns
import data_split
# import signal
# import resource
# from disc import Discriminator
# from torch_geometric.utils import negative_sampling, train_test_split_edges
# from torch_geometric.nn import
import metric
from torch.optim import lr_scheduler
from gae import GAE, VGAE
from LP_utils import *
import torch.utils.data as Data
import random
import copy


os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
sys.path.append("..")
sys.path.append("...")



parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='cora', help='Dataset to train, citeseer')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learing rate')
parser.add_argument('--epoches', type=int, default=200, help='Number of training epoches')
parser.add_argument('--hidden_dim', type=list, default=32, help='Dimensions of hidden layers')
parser.add_argument('--out_dim', type=list, default=16, help='Dimensions of out layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep  probability)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for l2 loss on embedding matrix')
# parser.add_argument('--X_NUM', type=int, default=716, help='Number of traing features')
parser.add_argument('--seed', type=int, default=30, help='Random seed.')#15 20 30 35                   #cora 30 citeseer 10 polblogs 30 cora_ml 35 pubmed 35    #cora 35 citeseer 10 polblogs 35 cora_ml 35 pubmed 35
parser.add_argument('--p', type=float, default=0.5, help='Hold data p.')
parser.add_argument('--q', type=float, default=0.5, help='Hold data q.')
parser.add_argument('--model', type=str, default='GCN', help='LP model.')
parser.add_argument('--eps', type=float, default=0.01, help='Value of epsilon.')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features, idx_train, idx_val, idx_test, labels = load_data1(args.datasets)
# target_nodes = np.array(random.sample(list(idx_test), 30))
if args.datasets == 'cora_ml' or 'amazon':
    idx_test = idx_test[:1000]
print(adj.sum()//2)
target_nodes = idx_test
A = np.array(adj.todense())



A_A, A_B, X_A, X_B = data_split.split_graph(args, adj, features, split_method='com', with_s=True, with_f=True)

A_A_copy = A_A.copy()
A_B_copy = A_B.copy()
X_A_copy = X_A.copy()
X_B_copy = X_B.copy()

#
A_A = normalize_adj(A_A)
A_B = normalize_adj(A_B)
A_A = sparse_mx_to_torch_sparse_tensor(A_A).to(device)
A_B = sparse_mx_to_torch_sparse_tensor(A_B).to(device)
X_A = sparse_mx_to_torch_sparse_tensor(X_A).to(device)
X_B = sparse_mx_to_torch_sparse_tensor(X_B).to(device)


X = sparse_to_tuple(features.tocoo())

models = {

    "client_A": GCN(nfeat=X_A.shape[1], nclass=args.out_dim, nhid=args.hidden_dim, device=device,dropout=0),
    "client_B": GCN(nfeat=X_B.shape[1], nclass=args.out_dim, nhid=args.hidden_dim, device=device,dropout=0),
    "server": nn.Sequential(
        # nn.ReLU(),
        nn.Linear(2*args.out_dim, int(labels.max()+1)),
    )
}
models['client_A'].to(device)
models['client_B'].to(device)
models['server'].to(device)


adj = normalize_adj(adj)
adj = sparse_mx_to_torch_sparse_tensor(adj)
features = sparse_mx_to_torch_sparse_tensor(features)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
labels = torch.LongTensor(labels)
if args.cuda:
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

total_grad_out = []
total_grad_in = []



def train(XA, AA, XB, AB):
    optimizer = optim.Adam(params=chain(models['client_A'].parameters(), models['client_B'].parameters(),
                                        models['server'].parameters()), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(params=[{'params':models['server'].parameters()},{'params':models['client_A'].parameters()},{'params':models['client_B'].parameters()}], lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epoches):
        models['client_A'].train()
        models['client_B'].train()
        models['server'].train()

        client_out = {}

        # for opt in optimizers:
        #     opt.zero_grad()
        optimizer.zero_grad()
        client_out['client_A'] = models['client_A'](XA, AA)
        client_out['client_B'] = models['client_B'](XB, AB)


        server_input = torch.cat((client_out['client_A'], client_out['client_B']), 1)

        if epoch == 0:
            pred = models['server'].to(device)(server_input)
        else:
            pred = models['server'](server_input)

        pred = F.log_softmax(pred, dim=-1)

        loss = F.nll_loss(pred[idx_train], labels[idx_train])
        predA = F.log_softmax(client_out['client_A'], dim=-1)
        predB = F.log_softmax(client_out['client_B'], dim=-1)

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print("Epoch: %d, train loss: %f"%(epoch, loss.cpu().detach().numpy()))


        if epoch == args.epoches-1:
            global save_outputpred
            save_outputpred = pred
            global save_outputA
            save_outputA = predA
            global save_outputB
            save_outputB = predB
            global emb_A
            emb_A = client_out['client_A']
            global emb_B
            emb_B = client_out['client_B']

            save_outputpred = Variable(save_outputpred, requires_grad=True)
            save_outputA = Variable(save_outputA, requires_grad=True)
            save_outputB = Variable(save_outputB, requires_grad=True)
            emb_A = Variable(emb_A, requires_grad=True)
            emb_B = Variable(emb_B, requires_grad=True)

            # save_predA = Variable(save_predA, requires_grad=True)
            torch.save(models['server'],'save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/server.pkl')
            torch.save(models['client_A'], 'save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/A.pkl')
            torch.save(models['client_B'], 'save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/B.pkl')
        # print(models[server.id].state_dict())

    return emb_A, emb_B, save_outputA, save_outputB, save_outputpred


G = nn.Sequential(
    nn.Linear(2*args.out_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, args.out_dim)
)
G = G.to(device)


def generate_emb(noise, emb_B, pred_S, D):
    optimizer1 = optim.Adam(params=G.parameters(), lr=0.01, weight_decay=args.weight_decay)
    D.eval()
    cat_emb = torch.cat((noise, emb_B), dim=-1)
    epochs = 2000
    for epoch in range(epochs):
        G.train()
        input_emb = G(cat_emb)
        optimizer1.zero_grad()
        # if epoch == 0:
        #     pred = D.to(device)(input_emb)
        # else:
        input_emb = torch.cat((input_emb, emb_B),dim=1)
        pred = D(input_emb)
        pred = F.log_softmax(pred, dim=-1)
        loss = F.mse_loss(pred, pred_S)
        loss.backward()
        optimizer1.step()
        if epoch % 10 == 0:
            print("Epoch: %d, train loss: %f"%(epoch, loss.cpu().detach().numpy()))
        if epoch == epochs-1:
            global server_gen_pred
            server_gen_pred = pred
            global embed_gen_A
            embed_gen_A = input_emb.split(noise.shape[1], dim=1)
    # print('Accuracy of generate_emb:', accuracy(server_gen_pred[idx_test], labels[idx_test]).item())
    return server_gen_pred, F.log_softmax(embed_gen_A[0],dim= 1), embed_gen_A[0]


def evasion_attack(XA, AA, XB, AB, target_node):
    rebuild_model = {
        'A': torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/A.pkl'), 'B': torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/B.pkl'),
        'S': torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/server.pkl')
    }
    rebuild_model['A'] = rebuild_model['A'].to(device)
    rebuild_model['B'] = rebuild_model['B'].to(device)
    rebuild_model['S'] = rebuild_model['S'].to(device)

    rebuild_model['A'].eval()
    rebuild_model['B'].eval()
    rebuild_model['S'].eval()
    client_out = {}

    client_out['client_A'] = rebuild_model['A'](XA, AA)
    client_out['client_B'] = rebuild_model['B'](XB, AB)
    server_input = torch.cat((client_out['client_A'], client_out['client_B']), 1)
    pred = rebuild_model['S'](server_input)

    outputA = F.log_softmax(client_out['client_A'], dim=-1)
    outputB = F.log_softmax(client_out['client_B'], dim=-1)
    outputS = F.log_softmax(pred, dim=-1)

    return accuracy(outputA[[target_node]], labels[[target_node]]).item(), accuracy(outputB[[target_node]], labels[
        [target_node]]).item(), accuracy(outputS[[target_node]], labels[[target_node]]).item(), torch.exp(outputS[[target_node]]), F.nll_loss(outputS[[target_node]],labels[[target_node]]).item(),pred[[target_node]]

def model_test(XA, AA, XB, AB, target_node):
    rebuild_model = {
        'A': torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/A.pkl'), 'B': torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/B.pkl'),
        'S': torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/server.pkl')
    }
    rebuild_model['A'] = rebuild_model['A'].to(device)
    rebuild_model['B'] = rebuild_model['B'].to(device)
    rebuild_model['S'] = rebuild_model['S'].to(device)

    rebuild_model['A'].eval()
    rebuild_model['B'].eval()
    rebuild_model['S'].eval()
    client_out = {}

    client_out['client_A'] = rebuild_model['A'](XA, AA)
    client_out['client_B'] = rebuild_model['B'](XB, AB)
    server_input = torch.cat((client_out['client_A'], client_out['client_B']), 1)
    pred = rebuild_model['S'](server_input)

    outputA = F.log_softmax(client_out['client_A'], dim=-1)
    outputB = F.log_softmax(client_out['client_B'], dim=-1)
    outputS = F.log_softmax(pred, dim=-1)

    return accuracy(outputA[[target_node]], labels[[target_node]]).item(), accuracy(outputB[[target_node]], labels[
        [target_node]]).item(), accuracy(outputS[[target_node]], labels[[target_node]]).item(), torch.exp(outputS[[target_node]]), F.nll_loss(outputS[[target_node]],labels[[target_node]]).item(),pred[[target_node]]

surrogate_server =  nn.Sequential(
        nn.Linear(2*args.out_dim, int(labels.max()+1)),
    )

surrogate_server = surrogate_server.to(device)
def surrogate_server_train(emb_infer, emb_B, pred_S):
    optimizer1 = optim.Adam(params=surrogate_server.parameters(), lr=0.01, weight_decay=args.weight_decay)
    cat_emb = torch.cat((emb_infer, emb_B), dim=-1)
    cat_emb = cat_emb.cpu().detach()
    cat_emb = cat_emb.to(device)
    epochs = 200
    for epoch in range(epochs):
        optimizer1.zero_grad()
        pred = surrogate_server(cat_emb)
        pred = F.log_softmax(pred, dim=-1)
        loss = F.mse_loss(pred, pred_S)
        loss.backward()
        optimizer1.step()
        if epoch % 10 == 0:
            print("Epoch: %d, train loss: %f"%(epoch, loss.cpu().detach().numpy()))
    torch.save(surrogate_server,'save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/surrogate_server.pkl')


def fgsm_attack(emb_infer, emb_B, surrogate_server, label_B, target_node, eps=0.5):
    cat_emb = torch.cat((emb_infer, emb_B), dim=-1)
    cat_emb = cat_emb.cpu().detach()
    cat_emb = cat_emb.to(device)
    base_emb = cat_emb[target_node].clone().cpu().data.numpy()
    input = cat_emb[target_node].clone()
    input.requires_grad=True


    # emb_B_target = emb_B[target_node]
    # emb_B_target.requires_grad=True
    # input = torch.cat((emb_infer[target_node],emb_B_target),dim=-1)

    surrogate_server.eval()
    pred = surrogate_server(input)
    surrogate_server.zero_grad()
    pred = F.log_softmax(pred, dim=-1)
    loss_target_node = F.nll_loss(pred, label_B[target_node])
    # loss_target_node = F.nll_loss(pred[idx_train], label_B[idx_train])
    loss_target_node.backward()
    grad_sign = input.grad.data.cpu().sign().numpy()
    emb_infer_with_noise = base_emb + eps*grad_sign
    input1 = torch.FloatTensor(emb_infer_with_noise).to(device)
    pred_pert = surrogate_server(input1)
    pred_pert = F.log_softmax(pred_pert, dim=-1)
    k_i = np.argmax(pred_pert.cpu().data.numpy())
    acc = 0
    label_target = label_B[target_node]
    if label_target!=k_i:
        acc+=1
    # print(cat_emb[target_node])
    # print(emb_infer_with_noise)
    return emb_infer_with_noise, acc


def inverse_attack(model_B, emb_infer_with_noise, target_node, A_B, X_B):
    t = target_node[0]
    # emb_B_noise = emb_infer_with_noise[:,:int(labels.max()+1)]
    # emb_B_noise = emb_infer_with_noise[:, :args.out_dim]
    emb_B_noise = emb_infer_with_noise[:,args.out_dim:2*args.out_dim]
    # emb_B_noise = 1/2*(emb_A_noise + emb_B_noise)

    emb_B_noise = torch.FloatTensor(emb_B_noise).to(device)
    model_B.eval()
    A_B = torch.FloatTensor(A_B.todense()).to(device)
    A_B.requires_grad=True
    adj_B = normalize_adj_tensor(A_B, sparse=False)
    emb = model_B(X_B, adj_B)
    model_B.zero_grad()
    loss = -F.mse_loss(emb[target_node], emb_B_noise)+1
    # loss.backward()
    # grad = A_B.grad.data.cpu().numpy()
    # grad = 0.5*(grad+grad.transpose())
    grad = torch.autograd.grad(loss, A_B)[0].data.cpu().numpy()
    grad = (grad[target_node] + grad[:, target_node])
    idx = grad[t].argmax()
    adversarial_edge = []
    adversarial_edge.append(np.array([t, idx]))
    return adversarial_edge


surrogate = GCN(nfeat=X_B_copy.shape[1], nclass=args.out_dim, nhid=args.hidden_dim, device=device)
surrogate = surrogate.to(device)

def train_surrogate(A, X, labels_B):

    A = normalize_adj(A)
    A = sparse_mx_to_torch_sparse_tensor(A)
    A = A.to(device)
    X = sparse_mx_to_torch_sparse_tensor(X)
    X = X.to(device)

    optimizer_s = optim.Adam(params=surrogate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epoches):
        surrogate.train()
        t = time.time()
        optimizer_s.zero_grad()
        output = surrogate(X, A)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output[idx_train], labels_B[idx_train])
        acc_train1 = accuracy(output[idx_train], labels_B[idx_train])
        loss.backward()
        optimizer_s.step()
        loss_val1 = F.nll_loss(output[idx_val], labels_B[idx_val])
        acc_val1 = accuracy(output[idx_val], labels_B[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.data.item()),
              'acc_train: {:.4f}'.format(acc_train1.data.item()),
              'loss_val: {:.4f}'.format(loss_val1.data.item()),
              'acc_val: {:.4f}'.format(acc_val1.data.item()),
              'time: {:.4f}s'.format(time.time() - t))



def cal_multi(y_true, y_pred, p_true, p_pred, logloss):
    """
    :param y_true: 真实类标
    :param y_pred: 预测类标
    :return: FPR、、、、、
    """
    p_pred = np.squeeze(p_pred, axis=1)
    Prcesion = sklearn.metrics.precision_score(y_true,y_pred,average='macro')
    F1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    Recall = sklearn.metrics.recall_score(y_true,y_pred,average='macro')
    ACC = sklearn.metrics.accuracy_score(y_true,y_pred)
    mae = sklearn.metrics.mean_absolute_error(p_true, p_pred)
    return ACC, Prcesion, Recall, F1, mae, logloss.mean()


def convert_to_one_hot(y, depth):
    return np.eye(depth)[y.reshape(-1)].T

def to_categorical(labels, num_classes, axis=0):
    ohe_labels = np.zeros((len(labels), num_classes)) if axis != 0 else np.zeros((num_classes, len(labels)))
    for _ in range(len(labels)):
        if axis == 0:
            ohe_labels[labels[_], _] = 1
        else:
            ohe_labels[_, labels[_]] = 1
    return ohe_labels

if __name__ == '__main__':
    # step1
    emb_A, emb_B, pred_A, pred_B, pred_S = train(X_A, A_A, X_B, A_B)
    labels_B = pred_S.max(1)[1].type_as(labels)
    torch.save(emb_A,'save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/emb_A.pth')
    torch.save(emb_B, 'save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/emb_B.pth')
    torch.save(labels_B, 'save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/labels_B.pth')
    # stealing embedding
    D = torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/server.pkl')
    D = D.to(device)
    noise = torch.rand([X_A.shape[0], args.out_dim]).to(device)
    t1 = time.time()
    gen_pred_S, gen_pred_A, gen_emb_A = generate_emb(noise, emb_B, pred_S, D)
    print("######################time_GRN######################",time.time()-t1)
    t2 = time.time()
    surrogate_server_train(gen_emb_A, emb_B, pred_S)
    print("######################time_shadow######################", time.time()-t2)
    torch.save(gen_emb_A, 'save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/gen_emb_A.pth')
    print('Accuracy of server:', accuracy(pred_S[idx_test], labels[idx_test]).item())

    model_A = torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/A.pkl')
    model_A.eval()
    out1 = model_A(X_A, A_A)
    pred_A = F.log_softmax(out1, dim=-1)


    model_B = torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/B.pkl')
    model_B = model_B.to(device)
    model_B.eval()
    out = model_B(X_B, A_B)
    pred_B = F.log_softmax(out, dim=-1)


    model_S = torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/server.pkl')
    model_S.eval()
    emb_S = torch.cat((out1, out), dim=-1)
    out2 = model_S(emb_S)
    pred_Server = F.log_softmax(out2, dim=-1)

    print('Accuracy of S:', accuracy(pred_Server[idx_test], labels[idx_test]).item())



    #step2
    gen_emb_A = torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/gen_emb_A.pth')
    labels_B = torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/labels_B.pth')
    emb_A = torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/emb_A.pth')
    emb_B = torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/emb_B.pth')
    emb = torch.cat((gen_emb_A, emb_B), dim=-1)

    s_server = torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/surrogate_server.pkl') #surrogate_server
    s_server = s_server.to(device)


    i = 0
    n_perturbations = 1
    cnt_A = 0
    cnt_B = 0
    cnt_S = 0
    t_A = 0
    t_B = 0
    t_S = 0
    y_predict_no_attack = []
    y_predict_attack = []
    p_predict_no_attack = []
    p_predict_attack = []
    label_target = []
    logloss_no_attack = []
    logloss_attack = []
    pred_ori = []
    pred_ad = []
    t_atk = time.time()
    num_atk = 0
    for j in tqdm(target_nodes):
        num_atk+=1
        label_target.append(labels[j].item())
        target_node = []
        target_node.append(j)
        target_node = np.array(target_node)

        emb_infer_with_noise, if_succ = fgsm_attack(gen_emb_A, emb_B, s_server, labels_B, target_node, eps=args.eps)
        i += if_succ

        model_B = torch.load('save_model/main/'+str(args.model)+'/'+str(args.datasets)+str(args.seed)+'/B.pkl')

        adversarial_edge = inverse_attack(model_B, emb_infer_with_noise, target_node, A_B_copy, X_B)
        modified_adj = A_B_copy.copy().tolil()


        for edge in adversarial_edge:
            edge = edge.transpose()
            if modified_adj[edge[0], edge[1]] != 0:
                modified_adj[edge[0], edge[1]] = 0
                modified_adj[edge[1], edge[0]] = 0
            else:
                modified_adj[edge[0], edge[1]] = 1
                modified_adj[edge[1], edge[0]] = 1

        modified_adj = modified_adj.tocsr()
        modified_adj = normalize_adj(modified_adj)
        modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
        modified_adj = modified_adj.to(device)
        if (num_atk%100) == 0:
            print("############atk time###############", time.time()-t_atk)
        acc_A_test, acc_B_test, acc_S_test, output_S_ori, logloss_ori, pred = model_test(X_A, A_A, X_B, A_B, target_node)
        acc_A, acc_B, acc_S, output_S_attack, logloss_atk, pred_atk = evasion_attack(X_A, A_A, X_B, modified_adj, target_node)

        cnt_A += acc_A
        cnt_B += acc_B
        cnt_S += acc_S
        t_A += acc_A_test
        t_B += acc_B_test
        t_S += acc_S_test

        y_predict_no_attack.append(np.argmax(output_S_ori.cpu().detach().numpy(), axis=-1))
        y_predict_attack.append(np.argmax(output_S_attack.cpu().detach().numpy(), axis=-1))
        p_predict_no_attack.append(output_S_ori.cpu().detach().numpy())
        p_predict_attack.append(output_S_attack.cpu().detach().numpy())
        logloss_no_attack.append(logloss_ori)
        logloss_attack.append(logloss_atk)
        pred_ori.append(pred.cpu().detach().numpy())
        pred_ad.append(pred_atk.cpu().detach().numpy())


    print('Accuracy of server:', t_S / len(target_nodes))

    print('Accuracy of server under attack:', cnt_S / len(target_nodes))
    label_target = np.array(label_target)

    p_label_target = to_categorical(label_target, int(labels.max()+1), axis=1)
    ACC1, Prcesion1, Recall1, F11, MAE1, logloss1 = cal_multi(label_target, np.array(y_predict_no_attack),  p_label_target, np.array(p_predict_no_attack), np.array(logloss_no_attack))
    ACC2, Prcesion2, Recall2, F12, MAE2, logloss2 = cal_multi(label_target, np.array(y_predict_attack),  p_label_target, np.array(p_predict_attack), np.array(logloss_attack))
    print("Ori ---  ACC: ", ACC1, " Prcesion: ", Prcesion1, " Recall: ", Recall1, " F1: ", F11, " MAE: ", MAE1, " Log Loss: ", logloss1)
    print("Atk ---  ACC: ", ACC2, " Prcesion: ", Prcesion2, " Recall: ", Recall2, " F1: ", F12, " MAE: ", MAE2, " Log Loss: ", logloss2)

