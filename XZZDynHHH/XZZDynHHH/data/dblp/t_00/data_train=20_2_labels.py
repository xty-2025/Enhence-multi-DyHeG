import pickle as pkl

# train-val-test set is equal for all timestep

#Key: 4056, Value: {'node_id': 14468, 'subgraph_idx': {0: [], 1: [], 2: [], 3: [], 4: [], 5: [2170, 2350], 6: []}, 'label': 0, 'time_id': 5, 'dataset': 'test'}
with open('data_train=20.pkl', "rb") as f:
    node_info = pkl.load(f)

train_author_node = {}
for i in range(7):
    time_step = i
    train_author_node[i] = []

val_author_node = []
test_author_node = []
for key, value in node_info.items():
    if (value['dataset'] == 'train'):
        train_author_node[value['time_id']].append([value['node_id'], value['label']])
    elif (value['dataset'] == 'val'):
        val_author_node.append([value['node_id'], value['label']])
    elif (value['dataset'] == 'test'):
        test_author_node.append([value['node_id'], value['label']])

for i in range(7):
    array_name = f"author_label{str(i).zfill(2)}"
    array_name = []
    array_name.append(train_author_node[i])
    array_name.append(val_author_node)
    array_name.append(test_author_node)


    print(array_name)
    print(len(array_name))
    print(len(array_name[0]))
    print(len(array_name[1]))
    print(len(array_name[2]))
    output_path = f"data_train=20_labels{str(i).zfill(2)}.pkl"
    with open(output_path, "wb") as f:
        pkl.dump(array_name,f)
print("done")


# print(train_author_node)
# print(len(train_author_node))
# print(val_author_node)
# print(len(val_author_node))
# print(test_author_node)
# print(len(test_author_node))