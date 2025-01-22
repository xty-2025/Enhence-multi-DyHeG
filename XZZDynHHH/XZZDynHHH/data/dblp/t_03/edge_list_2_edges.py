from scipy.sparse import csr_matrix
import numpy as np
import pickle

total_node = 37791

edge_list = open('edge_list.txt', mode='r', encoding='utf‐8')
pa_row_indices = []
pa_col_indices = []
pt_row_indices = []
pt_col_indices = []
pc_row_indices = []
pc_col_indices = []

pa_edgenum, pc_edgenum, pt_edgenum = 0, 0, 0
#
for line in edge_list:
    line = line.split("\n")[0].strip()
    h_id, t_id = line.split(" ")
    h_id = int(h_id)
    t_id = int(t_id)
    if h_id >= 14475 and h_id <= 28850:
        if t_id <= 14474:
            pa_row_indices.append(h_id)
            pa_col_indices.append(t_id)
            pa_edgenum += 1
        elif t_id >= 28851 and t_id <= 37770:
            pt_row_indices.append(h_id)
            pt_col_indices.append(t_id)
            pt_edgenum += 1
        elif t_id >= 37771 and t_id <= 37790:
            pc_row_indices.append(h_id)
            pc_col_indices.append(t_id)
            pc_edgenum += 1

edge_list.close()

pa_row_indices = np.array(pa_row_indices)
pa_col_indices = np.array(pa_col_indices)
pt_row_indices = np.array(pt_row_indices)
pt_col_indices = np.array(pt_col_indices)
pc_row_indices = np.array(pc_row_indices)
pc_col_indices = np.array(pc_col_indices)

pa_existedge = np.ones(pa_edgenum)
pt_existedge = np.ones(pt_edgenum)
pc_existedge = np.ones(pc_edgenum)

pa_matrix = csr_matrix((pa_existedge, (pa_row_indices, pa_col_indices)), shape=(total_node, total_node))
ap_matrix = csr_matrix((pa_existedge, (pa_col_indices, pa_row_indices)), shape=(total_node, total_node))
pt_matrix = csr_matrix((pt_existedge, (pt_row_indices, pt_col_indices)), shape=(total_node, total_node))
tp_matrix = csr_matrix((pt_existedge, (pt_col_indices, pt_row_indices)), shape=(total_node, total_node))
pc_matrix = csr_matrix((pc_existedge, (pc_row_indices, pc_col_indices)), shape=(total_node, total_node))
cp_matrix = csr_matrix((pc_existedge, (pc_col_indices, pc_row_indices)), shape=(total_node, total_node))


edges = {}
edges['p-a'] = pa_matrix
edges['a-p'] = ap_matrix
edges['p-t'] = pt_matrix
edges['t-p'] = tp_matrix
edges['p-c'] = pc_matrix
edges['c-p'] = cp_matrix

print(edges)
output_path = "edges.pkl"
with open(output_path, "wb") as f:
    pickle.dump(edges,f)
print("done")


