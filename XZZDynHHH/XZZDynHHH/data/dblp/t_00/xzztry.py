import pickle as pkl
import torch
import networkx as nx
import numpy as np

# with open('node_features.pkl', "rb") as f:
#     graphs = pkl.load(f)
# print(graphs.shape)
# print(type(graphs))

# a = torch.load('node_features.pth')
# print(type(a))
# print(a.shape)
# print(a[1])


# with open('data_train_labels.pkl', "rb") as f:
#     graphs = pkl.load(f)
# print(graphs)
# print(len(graphs))
# print(len(graphs[0]))
# print(len(graphs[1]))
# print(len(graphs[2]))

#
# with open('apcpa_emb_array.pkl', "rb") as f:
#     graphs1 = pkl.load(f)
# print(graphs1)

# zeros_array = np.zeros((37771, 37771))
# adj = np.matrix(zeros_array)
# output_path = "xzz.pkl"
# with open(output_path, "wb") as f:
#     pkl.dump(adj,f)
# print("done")


x = torch.tensor([ 1, 3])
y = torch.full((2,4), False, dtype=bool)
print(y)
y[0][x] = True
print(y)





'''
# -*- coding: utf-8 -*-
import dgl
import tqdm
import os
import multiprocessing

num_workers = 4

def construct_graph():
    node_src = [1,2,0,3,4,5,6,7]
    node_dst = [2,0,1,1,1,6,7,8]
    data1 = (node_src,node_dst)
    data2 = (node_dst,node_src)

    hg = dgl.heterograph(
            {('paper','pa','author'):data1,
            ('author','ap','paper'):data2}
    )
    print(hg)
    return hg

def walk(args):
    G,walk_length, start_node, schema,num_walks_per_node = args
    traces, _ = dgl.sampling.random_walk(
        G, [start_node] * num_walks_per_node, metapath=schema * walk_length)
    return traces

#"paper - Author - paper " metapath sampling
def generate_metapath():

    # path = '../output'
    # output_path = open(os.path.join(path, "output_path_pb.txt"), "w")
    schema = ['pa' if i% 2==0 else 'ap' for i in range(4)]

    num_process = 4
    num_walks_per_node = 5
    walk_length = 10
    hg = construct_graph()

    index_paper_map = {1:'a',2:'b',0:'c',3:'d',4:'e',5:'f',6:'g',7:'h'}
    index_author_map = {2:'i',0:'j',1:'k',6:'l',7:'m',8:'n'}

    with multiprocessing.Pool(processes=num_process) as pool:
        iter = pool.imap(walk, ((hg,walk_length, node, schema,num_walks_per_node) for node in tqdm.trange(hg.number_of_nodes('paper'))),chunksize=128)
        #iter 中包含了num(authors)*num_walks_per_node条路径
        for idx,traces in enumerate(iter):
            for tr in traces:
                print(tr)
                print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
                result = parse_trace(tr,index_paper_map,index_author_map)
                print(result)
                print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
                # output_path.write(result+'\n')
    # output_path.close()

    # for paper_id in tqdm.trange(hg.number_of_nodes('paper')):
    #
    #     #采样num_walks_per_node条路径
    #     traces, _ = dgl.sampling.random_walk(
    #             hg, [paper_id] * num_walks_per_node, metapath=schema*walk_length)
    #     #写入文件
    #     for tr in traces:
    #         result = parse_trace(tr,index_paper_map,index_author_map)
    #         output_path.write(result+'\n')
    # output_path.close()

def parse_trace(trace, index_paper_map, index_author_map):
    s = []
    trace = trace.numpy()
    for index in range(trace.shape[0]):
        if index % 2 == 0: #paper
            s.append(index_paper_map[trace[index]])
        else:              #author
            s.append(index_author_map[trace[index]])
    return ','.join(s)

if __name__ == '__main__':
    generate_metapath()
'''

'''
edge_list = open('edge_list.txt', mode='r', encoding='utf‐8')

a_num = 0
pa_edgenum, pc_edgenum, pt_edgenum = 0, 0, 0
for line in edge_list:
    line = line.split("\n")[0].strip()
    h_id, t_id = line.split(" ")
    h_id = int(h_id)
    t_id = int(t_id)


    if t_id <= 14474:
        a_num +=1

    if h_id >= 14475 and h_id <= 28850:

        if t_id <= 14474:
            pa_edgenum += 1
        elif t_id >= 28851 and t_id <= 37770:
            pt_edgenum += 1
        elif t_id >=37771 and t_id <= 37790:

            pc_edgenum += 1
print(pa_edgenum, pc_edgenum, pt_edgenum)
print(a_num)
input(sssss)
'''


