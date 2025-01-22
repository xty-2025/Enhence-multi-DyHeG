import pickle as pkl

total_paper_node_set = []
total_author_node_set = []
total_term_node_set = []
total_conf_node_set = []
for i in range(7):

    path = f"t_{str(i).zfill(2)}/node_have_set{str(i).zfill(2)}.pkl"
    with open(path, "rb") as f:
        file = pkl.load(f)
    total_paper_node_set.append(file[0])
    total_author_node_set.append(file[1])
    total_term_node_set.append(file[2])
    total_conf_node_set.append(file[3])

total_node = {}
total_node['total_paper_node_set'] = total_paper_node_set
total_node['total_author_node_set'] = total_author_node_set
total_node['total_term_node_set'] = total_term_node_set
total_node['total_conf_node_set'] = total_conf_node_set

output_path = "total_node.pkl"
with open(output_path, "wb") as f:
    pkl.dump(total_node,f)