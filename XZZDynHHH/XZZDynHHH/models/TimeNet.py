import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
import numpy as np

from XZZDynHHH.models.utils import SpaSeqNetLast, seq2str
from misc import DummyArgs
from torch.fft import rfft, irfft, fft, ifft

from sklearn.metrics import f1_score, accuracy_score
from misc import EarlyStopping, move_to, cal_metric
# from sild.utils import COLLECTOR
from libwon.utils.collector import *
from tqdm import tqdm, trange
from utils import *


class SpecMask(nn.Module):
    def __init__(self, hid_dim, temporature, K_len) -> None:
        super().__init__()
        self.K_len = K_len
        self.node_spec_map = nn.Sequential(nn.Linear(self.K_len * hid_dim * 2, hid_dim * 2), nn.ReLU(),
                                           nn.Linear(hid_dim * 2, self.K_len))  # N, K x d x 2 - > N x f
        self.temporature = temporature
        self.K_len = K_len
        self.hid_dim = hid_dim
        self.spec_lin = nn.Sequential(
            nn.Linear(K_len * hid_dim * 2, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim)
        )

    def forward(self, specs):
        # specs [N, T, d]
        # learn causal and spurious masks
        spec_real, spec_imag = specs.real, specs.imag  # [N, K, d]
        spec_real_imag = torch.stack([spec_real, spec_imag], dim=-1)  # [N, K, d, 2]
        node_choice = self.node_spec_map(
            spec_real_imag.view(-1, self.K_len * self.hid_dim * 2))  # [N, K * d * 2] -> [N, K]

        cmask_ = torch.sigmoid(node_choice / self.temporature)
        smask_ = torch.sigmoid(- node_choice / self.temporature)  # [N, K]
        if len(COLLECTOR.cache.get('cmask0', [])) == len(COLLECTOR.cache.get('loss', [])):
            COLLECTOR.add('cmask0', seq2str(cmask_[0].detach().cpu().numpy()))
            COLLECTOR.add('smask0', seq2str(smask_[0].detach().cpu().numpy()))

        cmask = cmask_.unsqueeze(-1).expand_as(spec_imag)
        smask = smask_.unsqueeze(-1).expand_as(spec_imag)

        # filter in the spectral domain
        c_spec_real = spec_real * cmask  # [N, K, d] * [N, K, d]
        c_spec_imag = spec_imag * cmask  # [N, K, d] * [N, K, d]

        s_spec_real = spec_real * smask  # [N, K, d] * [N, K, d]
        s_spec_imag = spec_imag * smask  # [N, K, d] * [N, K, d]

        c_spec = torch.cat([c_spec_real, c_spec_imag], dim=-1).flatten(-2, -1)
        s_spec = torch.cat([s_spec_real, s_spec_imag], dim=-1).flatten(-2, -1)

        c_spec = self.spec_lin(c_spec)  # [N, d]
        s_spec = self.spec_lin(s_spec)  # [N, d]

        return c_spec, s_spec


class TimeNet(torch.nn.Module):
    def __init__(self, args):
        super(TimeNet, self).__init__()
        in_dim = args.nfeat
        hid_dim = 2 * args.nhid
        out_dim = 2 * args.nhid
        num_layers = args.n_layers
        time_length = args.length
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.time_length = time_length
        self.spec_len = time_length
        self.K_len = 1 + self.spec_len // 2
        self.args = args
        self.len_train = self.time_length - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        self.len = time_length
        self.earlystop = EarlyStopping('val_auc', mode="max", patience=args.patience)
        self.device = args.device
        self.metric = args.metric
        self.args = args
        self.linear = nn.Linear(args.nfeat, hid_dim, bias=False)



        # spatio model
        if args.backbone in 'GCN GIN GAT'.split():
            args.static_conv = args.backbone
            self.backbone = SpaSeqNetLast(args)
        else:
            raise NotImplementedError()

        # spectral model
        if args.spec_filt == "mask":
            self.spec_filt = SpecMask(hid_dim, args.temporature, self.K_len)
        else:
            raise NotImplementedError()

        # post gnn
        self.post_gnn = args.post_gnn
        # if args.post_gnn:
        #     self.nodeconv = NodeFormer(hid_dim, hid_dim, hid_dim, num_layers = args.post_gnn)

        # decoder
        self.cs_decoder = self.backbone.cs_decoder
        self.ss_decoder = self.backbone.ss_decoder

        self.ctype = args.ctype



    def forward(self, graph_feature):
        # dataset
        x = graph_feature
        self.x = [x for _ in range(self.time_length)] if len(x.shape) <= 2 else x
        feature = self.x
        cs, ss = self.get_final_emb(feature) #get_final_emb() need edge_list and feature, but now we have got node feature from HGSL, so the edge_list is not need
        return cs, ss


    def spectral_filter(self, z):
        if not self.args.use_filt:
            return [z[-1], z[-1]]
        # z [T, N, d]
        ctype = self.ctype
        time_len = z.shape[0]
        # transform into spectral domain
        z = torch.permute(z, (1, 0, 2))  # [N, T, d]
        specs = rfft(z, n=self.spec_len, norm="ortho", dim=1)  # [N, K, d]

        # learn causal and spurious masks
        c_spec, s_spec = self.spec_filt(specs)  # [N, d]

        if self.post_gnn:
            c_spec = self.nodeconv(c_spec)
            s_spec = self.nodeconv(s_spec)

        out = [c_spec, s_spec]
        return out

    def get_final_emb(self, feature):
        cs, ss = [], []
        for t in range(self.time_length):
            # jump the GCN/GAT
            # the shape must is T*node_num*nhid (need bianhuan)
            x_list = feature[:t + 1] #type is list,len is timestep T, shape is node_num * dim(128)
            x_list = [self.linear(x) for x in x_list] #128 to 16, type is list,len is timestep T, shape is node_num * nhid(16)
            x_list = torch.stack(x_list)#type is tensor, shape is T*node_num*nhid
            cz, sz = self.spectral_filter(x_list)  # [N, d] is node_num * nhid(16)
            cs.append(cz)
            ss.append(sz)
        cs = torch.stack(cs, dim=0)  # [T, N, d]
        ss = torch.stack(ss, dim=0)  # [T, N, d]
        return cs, ss





'''
    def train_epoch(self, data):
        self.train()

        cz, sz = self.get_final_emb(data['edge_index'][:self.len_train])
        loss = self.cal_loss([cz, sz], data)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    @torch.no_grad()
    def test_epoch(self, data):
        model = self
        model.eval()

        cz, sz = self.get_final_emb(data['edge_index'])
        embeddings = cz

        auc_list = []
        for t in range(self.len):
            node_mask = data['node_masks'][t]
            preds = model.cs_decoder(embeddings[t])
            preds = preds[node_mask]
            preds = preds.argmax(dim=-1).squeeze()

            target = data['y'][node_mask].squeeze()
            auc = cal_metric(preds, target, self.args)
            auc_list.append(auc)
        train = np.mean(auc_list[:self.len_train])
        val = np.mean(auc_list[self.len_train:self.len_train + self.len_val])
        test = np.mean(auc_list[self.len_train + self.len_val:])

        COLLECTOR.add(key='auc_list', value=auc_list)

        return train, val, test
'''


