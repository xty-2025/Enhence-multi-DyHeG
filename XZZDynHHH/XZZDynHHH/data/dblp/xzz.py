import pickle as pkl


with open('total_node.pkl', "rb") as f:
    node_info = pkl.load(f)
print(node_info)
total_paper_dic = []
total_author_dic = []
total_term_dic = []
total_conf_dic = []
for i in range(7):
    total_paper_dic += node_info['total_paper_node_set'][i]
    total_author_dic += node_info['total_author_node_set'][i]
    total_term_dic += node_info['total_term_node_set'][i]
    total_conf_dic += node_info['total_conf_node_set'][i]

unique_elements = set(total_paper_dic)
total_paper_dic = list(unique_elements)
unique_elements = set(total_author_dic)
total_author_dic = list(unique_elements)
unique_elements = set(total_term_dic)
total_term_dic = list(unique_elements)
unique_elements = set(total_conf_dic)
total_conf_dic = list(unique_elements)
print(total_paper_dic, total_author_dic, total_term_dic, total_conf_dic)
