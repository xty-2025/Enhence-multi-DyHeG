

import pickle

meta_data = {}
meta_data['t_info'] = {'a': {'ind': range(0, 14475), 'cnt': 14475}, 'p': {'ind': range(14475, 28851), 'cnt': 14376},
                       't': {'ind': range(28851, 37771), 'cnt': 8920}, 'c': {'ind': range(37771, 37791), 'cnt': 20}}
meta_data['rel2id'] = {'p-a': 0, 'a-p': 1, 'p-t': 2, 't-p': 3, 'p-c': 4, 'c-p': 5}
meta_data['id2rel'] = {0: 'p-a', 1: 'a-p', 2: 'p-t', 3: 't-p', 4: 'p-c', 5: 'c-p'}
node2lid = {}
for i in range(0, 14475):
    key = 'a'+ str(i)
    node2lid[key] = i
for i in range(14475, 28851):
    key = 'p' + str(i-14475)
    node2lid[key] = i
for i in range(28851, 37771):
    key = 't' + str(i - 28851)
    node2lid[key] = i
for i in range(37771, 37791):
    key = 'c' + str(i - 37771)
    node2lid[key] = i
meta_data['node2lid'] = node2lid
    # input(fff)
node2gid = {}
for i in range(0, 14475):
    key = 'a'+ str(i)
    node2gid[key] = i
for i in range(14475, 28851):
    key = 'p' + str(i)
    node2gid[key] = i
for i in range(28851, 37771):
    key = 't' + str(i)
    node2gid[key] = i
for i in range(37771, 37791):
    key = 'c' + str(i)
    node2gid[key] = i
meta_data['node2gid'] = node2gid
gid2node={}
for key, value in node2gid.items():
    gid2node[value] = key
meta_data['gid2node'] = gid2node

meta_data['n_class'] = 4
meta_data['n_feat'] = 128 #initial node feature dim
meta_data['relations'] = ['p-a', 'a-p',  'p-t', 't-p', 'p-c', 'c-p']
meta_data['types'] = ['a', 'p', 't', 'c']
meta_data['undirected_relations'] = {'p-a', 'p-t', 'p-c'}
meta_data['r_info'] = {'p-a': (14475, 28851, 0, 14475), 'a-p': (0, 14475, 14475, 28851), 'p-t': (14475, 28851, 28851, 37771),
                       't-p': (28851, 37771, 14475, 28851), 'p-c': (14475, 28851, 37771, 37791), 'c-p': (37771, 37791, 14475, 28851)}

output_path = "meta_data.pkl"
with open(output_path, "wb") as f:
    pickle.dump(meta_data, f)

print("done")



