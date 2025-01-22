import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from util_funcs import cos_sim


class StrucNet(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, cf, g):
        super(StrucNet, self).__init__()
        self.__dict__.update(cf.get_model_conf())
        # ! Init variables
        self.dev = cf.dev
        self.ti, self.ri, self.types, self.ud_rels = g.t_info, g.r_info, g.types, g.undirected_relations# self.ri: {'p-a': (2957, 7285, 0, 2957), 'a-p': (0, 2957, 2957, 7285), 'p-c': (2957, 7285, 7285, 7305), 'c-p': (7285, 7305, 2957, 7285)}
        feat_dim, mp_emb_dim = g.features.shape[1], list(g.mp_emb_dict.values())[0].shape[1]
        self.non_linear = nn.ReLU()
        # ! Graph Structure Learning
        MD = nn.ModuleDict
        self.fgg_direct, self.fgg_left, self.fgg_right, self.fg_agg, self.sgg_gen, self.sg_agg, self.overall_g_agg = \
            MD({}), MD({}), MD({}), MD({}), MD({}), MD({}), MD({})# fg: feature graph sg: semantic graph
        # Feature encoder
        self.encoder = MD(dict(zip(g.types, [nn.Linear(g.features.shape[1], cf.com_feat_dim) for _ in g.types])))

        for r in g.undirected_relations:# undirected_relations: {'p-a', 'p-c'}
            # ! Feature Graph Generator
            self.fgg_direct[r] = GraphGenerator(cf.com_feat_dim, cf.num_head, cf.fgd_th, self.dev)
            self.fgg_left[r] = GraphGenerator(feat_dim, cf.num_head, cf.fgh_th, self.dev)
            self.fgg_right[r] = GraphGenerator(feat_dim, cf.num_head, cf.fgh_th, self.dev)
            self.fg_agg[r] = GraphChannelAttLayer(3)  # 3 = 1 (first-order/direct) + 2 (second-order)

            # ! Semantic Graph Generator
            self.sgg_gen[r] = MD(dict(
                zip(cf.mp_list, [GraphGenerator(mp_emb_dim, cf.num_head, cf.sem_th, self.dev) for _ in cf.mp_list])))
            self.sg_agg[r] = GraphChannelAttLayer(len(cf.mp_list))

            # ! Overall Graph Generator
            self.overall_g_agg[r] = GraphChannelAttLayer(3, [1, 1, 10])  # 3 = feat-graph + sem-graph + ori_graph
            # self.overall_g_agg[r] = GraphChannelAttLayer(3)  # 3 = feat-graph + sem-graph + ori_graph

        # ! Graph Convolution
        if cf.conv_method == 'gcn':
            self.GCN = GCN(g.n_feat, cf.emb_dim, g.n_class, cf.dropout)
        self.norm_order = cf.adj_norm_order

    def forward(self, group):
        features, adj_ori, mp_emb = group
        def get_rel_mat(mat, r):
            return mat[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]]

        def get_type_rows(mat, type):
            return mat[self.ti[type]['ind'], :]

        def gen_g_via_feat(graph_gen_func, mat, r):
            return graph_gen_func(get_type_rows(mat, r[0]), get_type_rows(mat, r[-1]))

        # ! Heterogeneous Feature Mapping
        com_feat_mat = torch.cat([self.non_linear(
            self.encoder[t](features[self.ti[t]['ind']])) for t in self.types]) #self.types: ['a', 'p', 'c'] ti: {'a': {'ind': range(0, 2957), 'cnt': 2957}, 'p': {'ind': range(2957, 7285), 'cnt': 4328}, 'c': {'ind': range(7285, 7305), 'cnt': 20}}

        # ! Heterogeneous Graph Generation
        new_adj = torch.zeros_like(adj_ori).to(self.dev) #7305*7305
        for r in self.ud_rels: # undirected_relations: {'p-a', 'p-c'}

            ori_g = get_rel_mat(adj_ori, r) # it is the matrix under the "r" relation

            # ! Feature Graph Generation
            fg_direct = gen_g_via_feat(self.fgg_direct[r], com_feat_mat, r)#generate the feature similarity graph
            fmat_l, fmat_r = features[self.ti[r[0]]['ind']], features[self.ti[r[-1]]['ind']]#ti is the domcu of range for each node type index
            sim_l, sim_r = self.fgg_left[r](fmat_l, fmat_l), self.fgg_right[r](fmat_r, fmat_r)# coculate the feature sim, this is the equ(4) of paper
            fg_left, fg_right = sim_l.mm(ori_g), sim_r.mm(ori_g.t()).t()# is the equ(5)(6)
            feat_g = self.fg_agg[r]([fg_direct, fg_left, fg_right])# is the equ(7) S(FS) S(FPH) S(FPT)

            # ! Semantic Graph Generation
            sem_g_list = [gen_g_via_feat(self.sgg_gen[r][mp], mp_emb[mp], r) for mp in mp_emb]#generate the semantic similarity graph equ(8)
            sem_g = self.sg_agg[r](sem_g_list)#use multi channel attention based on multi metepath equ(9)

            # ! Overall Graph
            # Update relation sub-matixs
            new_adj[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]] = \
                self.overall_g_agg[r]([feat_g, sem_g, ori_g])  # update edge  e.g. AP // self.ri[r][0]:self.ri[r][1] is the index range

        new_adj = new_adj + new_adj.t()  # sysmetric
        # ! Aggregate
        new_adj = F.normalize(new_adj, dim=0, p=self.norm_order)
        logits = self.GCN(features, new_adj)
        return logits, new_adj


class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        return h * self.weight


class GraphGenerator(nn.Module):
    """
    Generate graph using similarity.
    """

    def __init__(self, dim, num_head=2, threshold=0.1, dev=None):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
        self.num_head = num_head
        self.dev = dev

    def forward(self, left_h, right_h):
        """

        Args:
            left_h: left_node_num * hidden_dim/feat_dim
            right_h: right_node_num * hidden_dim/feat_dim
        Returns:

        """
        if torch.sum(left_h) == 0 or torch.sum(right_h) == 0:
            return torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        s = torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        zero_lines = torch.nonzero(torch.sum(left_h, 1) == 0)
        # The ReLU function will generate zero lines, which lead to the nan (devided by zero) problem.
        if len(zero_lines) > 0:
            left_h[zero_lines, :] += 1e-8
        for i in range(self.num_head):
            weighted_left_h = self.metric_layer[i](left_h)
            weighted_right_h = self.metric_layer[i](right_h)
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_head
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
        # return x

# this is the channel attention layer
class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # equal weight init it by 0.1 of each member
        # if weights != None:
        #     # self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
        #     with torch.no_grad():
        #         w = torch.Tensor(weights).reshape(self.weight.shape)
        #         self.weight.copy_(w)

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)# stack it to tensor
        # Row normalization of all graphs generated
        adj_list = F.normalize(adj_list, dim=1, p=1)# guiyihua
        # Hadamard product + summation -> Conv
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)#sum the weighted num at each channel

class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output
