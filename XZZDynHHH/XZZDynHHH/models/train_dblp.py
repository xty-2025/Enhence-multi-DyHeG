# -*- encoding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/11/19 17:14
@Author  :   Zhezhe Xing
@Contact :   zhezhexing123@163.com
JLU, Changchun, China
'''
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
from libwon.utils.collector import *
from misc import EarlyStopping, setup_seed, get_arg_dict, move_to

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
sys.path.append(root_path + 'src')
os.chdir(root_path)

from hin_loader import HIN

import util_funcs as uf
from config import StrucNetConfig
from StrucNet import StrucNet
from TimeNet import TimeNet

from DyHHH import DyHHH

import warnings
import time
import torch
import argparse

warnings.filterwarnings('ignore')
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]

torch.autograd.set_detect_anomaly(True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--log_on', type=bool, nargs='?', default=True)
    parser.add_argument('--time_steps', type=int, nargs='?', default=2,
                        help="total time steps used for train, eval and test")
    parser.add_argument('--author_index_max', type=int, nargs='?', default=14475,
                        help='the max index of author is 0-14474, but the known label author num is 4075')
    parser.add_argument('--node_index_max', type=int, nargs='?', default=37791,
                        help='the max index of node is 0-37790')
    parser.add_argument('--topk', type=int, nargs='?', default=10,
                        help="topk num")
    parser.add_argument('--total_node', type=int, nargs='?', default=-1,
                        help="total node")

    parser.add_argument('--datapath', type=str, nargs='?', default='../XZZDynHHH/data/beauty',
                        help='datapath')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=500,
                        help='# epochs')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                        help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                        help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=200,
                        help="patient")
    # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
    parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual')
    parser.add_argument('--gcnresidual', type=bool, nargs='?', default=False,
                        help='Use residual')
    # Number of negative samples per positive pair.
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive')
    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--GCN_dropout', type=float, default=0.5,
                        help='GCN_Dropout rate (1 - keep probability).')
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')

    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
    parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')

    #TimeNet
    # 1.dataset
    parser.add_argument('--dataset', type=str, default='dblp', help='datasets')
    parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
    parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
    parser.add_argument('--num_classes', type=int, default=-1, help='')
    parser.add_argument('--length', type=int, default=-1, help='')
    parser.add_argument('--testlength', type=int, default=0, help='length for test')
    parser.add_argument('--P', type=float, default=0.5)
    parser.add_argument('--SIGMA', type=float, default=0.3)
    parser.add_argument('--TEST_P', type=float, default=-0.8)
    parser.add_argument('--TEST_SIGMA', type=float, default=0.1)
    parser.add_argument('--use_cfg', type=int, default=1)

    # 1.5 exp misc
    parser.add_argument('--output_folder', type=str, default='', help='need to be modified')
    parser.add_argument('--log_dir', type=str, default="EXP")
    parser.add_argument('--log_interval', type=int, default=20, help='')

    # 2.experiments
    parser.add_argument('--max_epoch', type=int, default=1000, help='number of epochs to train.')
    parser.add_argument('--min_epoch', type=int, default=0, help='min epoch')
    parser.add_argument('--device', type=str, default='cpu', help='training device')
    parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
    parser.add_argument('--sampling_times', type=float, default=1, help='negative sampling times')

    # 3. params

    parser.add_argument('--nhid', type=int, default=8, help='dim of hidden embedding')  # 8
    parser.add_argument('--n_layers', type=int, default=2)

    parser.add_argument('--undirected', type=int, default=0)

    parser.add_argument('--attn_drop', type=float, default=0)
    parser.add_argument('--temp_attn_res', type=int, default=1)

    # 4. special
    parser.add_argument('--window_size', type=int, default=5, help='')
    parser.add_argument('--spec_len', type=int, default=-1)
    parser.add_argument('--mtype', type=int, default=0, help='')

    # type
    parser.add_argument('--ctype', type=int, default=0)
    parser.add_argument('--ltype', type=int, default=0, help='')
    parser.add_argument('--use_filt', type=int, default=1, help='')

    # searchable args
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
    parser.add_argument('--heads', type=int, default=4, help='attention heads.')  # 4
    parser.add_argument('--norm', type=int, default=1)  # define the type of Normalization
    parser.add_argument('--use_RTE', type=int, default=1, help='')
    parser.add_argument('--spec_RTE', type=int, default=1, help='')
    parser.add_argument('--spec_res', type=int, default=1, help='')

    parser.add_argument('--backbone', type=str, default="GAT")
    parser.add_argument('--static_conv', type=str, default="GCN")
    parser.add_argument('--spec_filt', type=str, default='mask')
    parser.add_argument('--post_temporal', type=int, default=0)
    parser.add_argument('--post_gnn', type=int, default=0)

    parser.add_argument('--temporature', type=float, default=1, help='')
    parser.add_argument('--intervene_lambda', type=float, default=0, help='')
    parser.add_argument('--intervene_times', type=int, default=100, help='')
    parser.add_argument('--learns', type=int, default=0)

    # 5. nodeclf
    parser.add_argument('--clf_layers', type=int, default=2, help='')
    parser.add_argument('--metric', type=str, default='acc')
    parser.add_argument('--main_metric', type=str, default='val_acc')

    # 6. dida
    parser.add_argument('--only_causal', type=int, default=0)
    parser.add_argument('--fmask', type=int, default=1)
    parser.add_argument('--lin_bias', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--skip', type=int, default=0)

    args = parser.parse_args()
    print(args)

setup_seed(args.seed)

args.length = 7
args.author_num_nodes = 4057
args.num_classes = 4
args.testlength = 0
args.vallength = 0



uf.seed_init(args.seed)
uf.shell_init(gpu_id=args.gpu_id)
cf = StrucNetConfig(args.dataset)  # load config info

# ! Modify config
cf.update(args.__dict__)
cf.dev = torch.device("cuda:0" if args.gpu_id >= 0 else "cpu")

# ! Load Graphs
data_bag = {}
for timestep in range (args.time_steps):
    each_data_bag = {}
    g = HIN(cf.dataset, timestep).load_mp_embedding(args, cf)  # load meta_path embedding apcpa_emb.pkl/aptpa_emb.pkl
    print(f'Dataset: {cf.dataset}, {g.t_info}')  # print g.t_info is the information about model train
    features, adj, mp_emb, train_x, train_y, val_x, val_y, test_x, test_y = g.to_torch(cf)
    each_data_bag['features'] = features
    each_data_bag['adjs'] = adj
    each_data_bag['mp_embs'] = mp_emb
    each_data_bag['train_x'] = train_x
    each_data_bag['train_y'] = train_y
    each_data_bag['val_x'] = val_x
    each_data_bag['val_y'] = val_y
    each_data_bag['test_x'] = test_x
    each_data_bag['test_y'] = test_y
    data_bag[timestep] = each_data_bag


# pre-logs
log_dir = args.log_dir
info_dict = get_arg_dict(args)
json.dump(info_dict,open(os.path.join(log_dir,'args.json'),'w'),indent=2)
print(args)

model = DyHHH(args, cf, data_bag, g).to(cf.dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
earlystop = EarlyStopping('val_auc', mode="max", patience = args.patience)
# in training
best_epoch_val = 0
patient = 0

for epoch in tqdm(range(args.epochs)):
    model.train()

    cz, sz = model(args, data_bag)
    # struc loss can be ignoaled or we can use the 'paper loss'!!!!!!!!!!

    # time loss
    loss = model.cal_loss([cz, sz], data_bag)
    loss.backward()
    opt.step()
    model.eval()

    final_cz, final_sz = model(args, data_bag)

    train_auc,val_auc,test_auc = model.evaluation(data_bag, [final_cz, final_sz])
    print("Epoch {:<3},  Loss = {:.3f}, Val AUC {:.3f} Test AUC {:.3f} ".format(epoch, loss, val_auc, test_auc))

    metrics = dict(zip('epoch,loss,train_auc,val_auc,test_auc'.split(','), [epoch, loss, train_auc, val_auc, test_auc]))
    for k, v in metrics.items():
        COLLECTOR.add(key=f'{k}', value=v)
    if earlystop.step(**metrics):
        break
best_metrics = earlystop.best_metrics
for k, v in best_metrics.items():
    COLLECTOR.add(key=f'best_{k}', value=v)
print(best_metrics)

COLLECTOR.add_GPU_MEM(args.device, id=False)
COLLECTOR.save_all_time()
COLLECTOR.save(os.path.join(log_dir,'collector.json'))

