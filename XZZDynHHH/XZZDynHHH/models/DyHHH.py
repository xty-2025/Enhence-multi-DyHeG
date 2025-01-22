
import argparse
import networkx as nx
import numpy as np
import pickle as pkl
import scipy
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
import sys
from early_stopper import *
from hin_loader import HIN
from libwon.utils.collector import *
from XZZDynHHH.models.utils import SpaSeqNetLast, seq2str
import util_funcs as uf
from config import StrucNetConfig
from StrucNet import StrucNet
from TimeNet import TimeNet
import warnings
import time
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from misc import EarlyStopping, move_to, cal_metric


class DyHHH(nn.Module):
    def __init__(self, args, cf, data_bag, g):
        super(DyHHH, self).__init__()
        self.args = args
        self.dev = cf.dev
        self.cf = cf
        self.g = g
        self.data_bag = data_bag
        self.G_emb, self.time_emb = self.build_model()
        # spatio model
        if args.backbone in 'GCN GIN GAT'.split():
            args.static_conv = args.backbone
            self.backbone = SpaSeqNetLast(args)
        else:
            raise NotImplementedError()
        self.cs_decoder = self.backbone.cs_decoder
        self.ss_decoder = self.backbone.ss_decoder
        if args.backbone in 'GCN GIN GAT'.split():
            args.static_conv = args.backbone
            self.backbone = SpaSeqNetLast(args)
        else:
            raise NotImplementedError()


    def forward(self, args, data_bag):#'data' or not 'data'(TimeNet), g(HGSL)

        #graph embedding
        G_emb_out = []
        for t in range(0, args.time_steps):
            features = data_bag[t]['features']
            adj_ori = data_bag[t]['adjs']
            mp_emb = data_bag[t]['mp_embs']
            logits, _ = self.G_emb([features, adj_ori, mp_emb])
            G_emb_out.append(logits)

        # time embedding
        cz, sz = self.time_emb(G_emb_out)
        return cz, sz


    def build_model(self):
        # 0: Graph Embedding Layer
        # generate each node feature of every timestep
        G_emb_layer = nn.Sequential()
        layer = StrucNet(cf = self.cf, g = self.g)
        G_emb_layer.add_module(name="G_emb_layer_{}".format(0), module=layer)

        #Time embedding
        Time_emb_layers = nn.Sequential()
        layer = TimeNet(args = self.args)
        Time_emb_layers.add_module(name="Time_emb_layer_{}".format(0), module=layer)

        return G_emb_layer, Time_emb_layers


    def cal_loss(self, zs, data_bag):#just use node_mask and y
        # node_masks is to mask the unknown label of paper's classes to control the supervise

        cs, ss = zs  # [T, N, F], [T, N, F] z starts from 0 to T
        ss = ss.detach()
        # get author node features
        cs_author = cs[:, 0: self.args.author_index_max, :]
        ss_ahthor = ss[:, 0: self.args.author_index_max, :]

        args = self.args
        device = args.device
        intervention_times, la = args.intervene_times, args.intervene_lambda

        def cal_y(embeddings, decoder, node_masks):
            preds = torch.tensor([]).to(device)
            for t in range(self.len_train):
                z = embeddings[t]
                mask = node_masks[t]
                pred = decoder(z)[mask]
                preds = torch.cat([preds, pred])
            return preds
        # define node_mask to control the supervise information

        # node_masks = data['node_masks'] #the sild model set the known mask infor
        # we firstly try to use all nodes to train (no mask)
        node_masks = []
        for t in range(args.time_steps):
            no_mask = torch.full((args.author_index_max), False, dtype=bool)
            node_masks.append(no_mask)
        # change the none label author node mask flag to 'false' let the node can't be used in comput, and take the val&test to 'false'
        for t in range(args.time_steps):
            node_masks[t][data_bag[t]['train_x']] = True # change the train node to True (for every timestep)


        node_labels = torch.LongTensor([]).to(device)
        for t in range(self.len_train):
            label = data_bag[t]['train_y'].squeeze().to(args.device)
            node_labels = torch.cat([node_labels, label])

        criterion = torch.nn.CrossEntropyLoss()
        cy = cal_y(cs, self.cs_decoder, node_masks)  # [N,C]
        sy = cal_y(ss, self.ss_decoder, node_masks)  # [N,C]

        conf_loss = criterion(sy, node_labels)
        COLLECTOR.add(key='conf_loss', value=conf_loss.item())
        causal_loss = criterion(cy, node_labels)
        COLLECTOR.add(key='causal_loss', value=causal_loss.item())

        if self.args.intervene_times <= 0 or self.args.intervene_lambda <= 0:
            loss = causal_loss
        else:
            env_loss = torch.tensor([]).to(device)
            for i in range(intervention_times):
                s1 = np.random.randint(len(sy))
                s = torch.sigmoid(sy[s1]).detach()
                conf = s * cy
                env_loss = torch.cat([env_loss, criterion(conf, node_labels).unsqueeze(0)])
            env_mean = env_loss.mean()
            env_var = torch.var(env_loss * intervention_times)
            penalty = env_mean + env_var
            loss = causal_loss + la * penalty
            COLLECTOR.add('env_mean', env_mean.item())
            COLLECTOR.add('env_var', env_var.item())

        loss = loss + conf_loss

        return loss

    def evaluation(self, data_bag, final_features):
        args = self.args

        cz, sz = final_features
        embeddings = cz

        # train auc

        # generate train node_mask, change to True means know label of the node
        # change the none label author node mask flag to 'false' let the node can't be used in comput, and take the val&test to 'false'
        train_node_masks = []
        for t in range(args.time_steps):
            no_mask = torch.full((args.author_index_max), False, dtype=bool)
            train_node_masks.append(no_mask)
        for t in range(args.time_steps):
            train_node_masks[t][data_bag[t]['train_x']] = True  # change the train node to True (for every timestep)
        # compute train auc
        train_auc_list = []
        for t in range(self.time_steps):
            train_node_mask = train_node_masks[t]
            train_preds = self.cs_decoder(embeddings[t])
            train_preds = train_preds[train_node_mask]
            train_preds = train_preds.argmax(dim=-1).squeeze()

            train_target = data_bag[t]['train_y'].squeeze()
            train_auc = cal_metric(train_preds, train_target, self.args)
            train_auc_list.append(train_auc)
        train_auc = np.mean(train_auc_list)

        # val auc

        # generate val node_mask, change to True means know label of the node
        val_node_masks = []
        for t in range(args.time_steps):
            no_mask = torch.full((args.author_index_max), False, dtype=bool)
            val_node_masks.append(no_mask)
        for t in range(args.time_steps):
            val_node_masks[t][data_bag[t]['train_x']] = True  # change the train node to True (for every timestep)
        #compute val auc
        val_auc_list = []
        for t in range(self.time_steps):
            val_node_mask = val_node_masks[t]
            val_preds = self.cs_decoder(embeddings[t])
            val_preds = val_preds[val_node_mask]
            val_preds = val_preds.argmax(dim=-1).squeeze()

            val_target = data_bag[t]['val_y'].squeeze()
            val_auc = cal_metric(val_preds, val_target, self.args)
            val_auc_list.append(val_auc)
        val_auc = np.mean(val_auc_list)

        # test auc

        # generate test node_mask, change to True means know label of the node
        test_node_masks = []
        for t in range(args.time_steps):
            no_mask = torch.full((args.author_index_max), False, dtype=bool)
            test_node_masks.append(no_mask)
        for t in range(args.time_steps):
            test_node_masks[t][data_bag[t]['train_x']] = True  # change the train node to True (for every timestep)
        # compute test auc
        test_auc_list = []
        for t in range(self.time_steps):
            test_node_mask = test_node_masks[t]
            test_preds = self.cs_decoder(embeddings[t])
            test_preds = test_preds[test_node_mask]
            test_preds = test_preds.argmax(dim=-1).squeeze()

            test_target = data_bag[t]['test_y'].squeeze()
            test_auc = cal_metric(test_preds, test_target, self.args)
            test_auc_list.append(test_auc)
        test_auc = np.mean(test_auc_list)


        COLLECTOR.add(key='train_auc_list', value=train_auc_list)
        COLLECTOR.add(key='val_auc_list', value=val_auc_list)
        COLLECTOR.add(key='test_auc_list', value=test_auc_list)

        return train_auc, val_auc, test_auc

