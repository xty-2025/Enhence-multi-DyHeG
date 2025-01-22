import os
import random
import sys
import time
import dgl
import numpy as np
import tqdm
import pickle as pkl


num_walks_per_node = 1000
walk_length = 100

def construct_graph():

    #relation type: paper-author paper-conference paper-term
    node_types = open('node_types.csv', mode='r', encoding='utf‐8')

    author_names, paper_names, term_names, conf_names = [], [], [], []

    # every type node index start 0
    author_map, paper_map, term_map, conf_map = {}, {}, {}, {}

    #num of each type node, as well as the node index
    a, p, t, c = 0, 0, 0, 0
    coun = 0
    for line in node_types:
        if(coun > 0):
            line = line.split("\n")[0].strip()
            node_id, node_type, _ = line.split(",")
            if (node_type == '0'):
                author_map[int(node_id)] = a
                author_names.append(node_id)
                a += 1
            if (node_type == '1'):
                paper_map[int(node_id)] = p
                paper_names.append(node_id)
                p += 1
            if (node_type == '2'):
                term_map[int(node_id)] = t
                term_names.append(node_id)
                t += 1
            if (node_type == '3'):
                conf_map[int(node_id)] = c
                conf_names.append(node_id)
                c += 1
        coun += 1
    print(a,p,t,c)

    # flag the node exist in the edge_list
    author_flag = np.zeros(a)
    paper_flag = np.zeros(p)
    term_flag = np.zeros(t)
    conf_flag = np.zeros(c)

    edge_list = open('edge_list.txt', mode='r', encoding='utf‐8')
    pa = []
    pc = []
    pt = []


    paper_have_dic = []
    author_have_dic = []
    term_have_dic = []
    conf_have_dic = []



    pa_edgenum, pc_edgenum, pt_edgenum = 0, 0, 0
    for line in edge_list:
        line = line.split("\n")[0].strip()
        h_id, t_id = line.split(" ")
        h_id = int(h_id)
        t_id = int(t_id)
        if h_id >= 14475 and h_id <= 28850:
            if t_id <= 14474:
                pa.append([paper_map[h_id], author_map[t_id]])
                if (paper_flag[paper_map[h_id]]) == 0:
                    paper_flag[paper_map[h_id]] = 1
                    paper_have_dic.append(paper_map[h_id])
                if(author_flag[author_map[t_id]]) ==0:
                    author_flag[author_map[t_id]] = 1
                    author_have_dic.append(author_map[t_id])
                pa_edgenum += 1
            elif t_id >= 28851 and t_id <= 37770:
                pt.append([paper_map[h_id], term_map[t_id]])
                if (paper_flag[paper_map[h_id]]) == 0:
                    paper_flag[paper_map[h_id]] = 1
                    paper_have_dic.append(paper_map[h_id])
                if (term_flag[term_map[t_id]]) == 0:
                    term_flag[term_map[t_id]] = 1
                    term_have_dic.append(term_map[t_id])
                pt_edgenum += 1
            elif t_id >=37771 and t_id <= 37790:
                pc.append([paper_map[h_id], conf_map[t_id]])
                if (paper_flag[paper_map[h_id]]) == 0:
                    paper_flag[paper_map[h_id]] = 1
                    paper_have_dic.append(paper_map[h_id])
                if (conf_flag[conf_map[t_id]]) == 0:
                    conf_flag[conf_map[t_id]] = 1
                    conf_have_dic.append(conf_map[t_id])
                pc_edgenum += 1
    print(pa_edgenum, pc_edgenum, pt_edgenum)

    node_have_set = []
    node_have_set.append(paper_have_dic)
    node_have_set.append(author_have_dic)
    node_have_set.append(term_have_dic)
    node_have_set.append(conf_have_dic)
    output_path = "node_have_set03.pkl"
    with open(output_path, "wb") as f:
        pkl.dump(node_have_set,f)



    edge_list.close()
    node_types.close()

    paper_author_src = []
    paper_author_dst = []
    paper_conf_src = []
    paper_conf_dst = []
    paper_term_src = []
    paper_term_dst = []

    for mem1 in pa:
        paper_author_src.append(mem1[0])
        paper_author_dst.append(mem1[1])
    for mem2 in pc:
        paper_conf_src.append(mem2[0])
        paper_conf_dst.append(mem2[1])
    for mem3 in pt:
        paper_term_src.append(mem3[0])
        paper_term_dst.append(mem3[1])

    hg = dgl.heterograph(
        {
            ("paper", "pa", "author"): (paper_author_src, paper_author_dst),
            ("author", "ap", "paper"): (paper_author_dst, paper_author_src),
            ("paper", "pc", "conf"): (paper_conf_src, paper_conf_dst),
            ("conf", "cp", "paper"): (paper_conf_dst, paper_conf_src),
            ("paper", "pt", "term"): (paper_term_src, paper_term_dst),
            ("term", "tp", "paper"): (paper_term_dst, paper_term_src),
        }
    )
    return hg, author_names, paper_names, term_names, conf_names, author_flag, paper_flag, term_flag, conf_flag


def generate_metapath_aptpa():
    output_path = open("aptpa_path.txt", "w")
    count = 0

    hg, author_names, paper_names, term_names, conf_names, author_flag, paper_flag, term_flag, conf_flag = construct_graph()

    s = 10
    author_idx_traces = {}
    for author_idx in tqdm.trange(hg.num_nodes("author")):#hg.num_nodes("conf") = 16
        # print([conf_idx] * num_walks_per_node): [0,0,0,0,0...]
        # print(["cp", "pa", "ap", "pc"] * walk_length): ['cp', 'pa', 'ap', 'pc', 'cp', 'pa', 'ap', 'pc', ...]

        if (author_flag[author_idx] == 0): # meaning the node not exist in the graph
            continue
        traces, _ = dgl.sampling.random_walk(
            hg,
            [author_idx] * num_walks_per_node,
            metapath=["ap", "pt", "tp", "pa"] * walk_length,
        )
        # author_idx_traces[author_idx] = traces

        s-=1
        m=10
        for tr in traces:
            m-=1
            selected_names = []

            for i in range(0, len(tr)):
                if (i % 4 == 0):
                    selected = author_names[tr[i]]
                if (i % 4 == 1):
                    selected = paper_names[tr[i]]
                if (i % 4 == 2):
                    selected = term_names[tr[i]]
                if (i % 4 == 3):
                    selected = paper_names[tr[i]]
                selected_names.append(selected)
            outline = " ".join(selected_names)
            print(outline, file=output_path)
        #     if m == 0:
        #         break
        # if s == 0:
        #     break
    # # print(author_idx_traces[0])
    # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
    # # print(author_idx_traces[1])
    # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
    # print(author_idx_traces[10])
    output_path.close()

def generate_metapath_apcpa():
    output_path = open("apcpa_path.txt", "w")
    count = 0

    hg, author_names, paper_names, term_names, conf_names, author_flag, paper_flag, term_flag, conf_flag = construct_graph()

    s = 10
    author_idx_traces = {}
    for author_idx in tqdm.trange(hg.num_nodes("author")):#hg.num_nodes("conf") = 16
        # print([conf_idx] * num_walks_per_node): [0,0,0,0,0...]
        # print(["cp", "pa", "ap", "pc"] * walk_length): ['cp', 'pa', 'ap', 'pc', 'cp', 'pa', 'ap', 'pc', ...]

        if (author_flag[author_idx] == 0): # meaning the node not exist in the graph
            continue
        traces, _ = dgl.sampling.random_walk(
            hg,
            [author_idx] * num_walks_per_node,
            metapath=["ap", "pc", "cp", "pa"] * walk_length,
        )
        # author_idx_traces[author_idx] = traces

        s-=1
        m=10
        for tr in traces:
            m-=1
            selected_names = []
            for i in range(0, len(tr)):
                if (i % 4 == 0):
                    selected = author_names[tr[i]]
                if (i % 4 == 1):
                    selected = paper_names[tr[i]]
                if (i % 4 == 2):
                    selected = conf_names[tr[i]]
                if (i % 4 == 3):
                    selected = paper_names[tr[i]]
                selected_names.append(selected)
            outline = " ".join(selected_names)
            print(outline, file=output_path)
        #     if m == 0:
        #         break
        # if s == 0:
        #     break
    # # print(author_idx_traces[0])
    # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
    # # print(author_idx_traces[1])
    # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
    # print(author_idx_traces[10])
    output_path.close()


def generate_metapath_pap():
    output_path = open("pap_path.txt", "w")
    count = 0

    hg, author_names, paper_names, term_names, conf_names, author_flag, paper_flag, term_flag, conf_flag = construct_graph()

    s = 10
    author_idx_traces = {}
    for paper_idx in tqdm.trange(hg.num_nodes("paper")):#hg.num_nodes("conf") = 16
        # print([conf_idx] * num_walks_per_node): [0,0,0,0,0...]
        # print(["cp", "pa", "ap", "pc"] * walk_length): ['cp', 'pa', 'ap', 'pc', 'cp', 'pa', 'ap', 'pc', ...]

        if (paper_flag[paper_idx] == 0): # meaning the node not exist in the graph
            continue
        traces, _ = dgl.sampling.random_walk(
            hg,
            [paper_idx] * num_walks_per_node,
            metapath=["pa", "ap"] * walk_length,
        )
        # author_idx_traces[author_idx] = traces

        s-=1
        m=10
        for tr in traces:
            m-=1
            selected_names = []
            for i in range(0, len(tr)):
                if (i % 2 == 0):
                    selected = paper_names[tr[i]]
                else:
                    selected = author_names[tr[i]]
                selected_names.append(selected)
            outline = " ".join(selected_names)
            print(outline, file=output_path)
        #     if m == 0:
        #         break
        # if s == 0:
        #     break
    # # print(author_idx_traces[0])
    # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
    # # print(author_idx_traces[1])
    # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
    # print(author_idx_traces[10])
    output_path.close()

def generate_metapath_ptp():
    output_path = open("ptp_path.txt", "w")
    count = 0

    hg, author_names, paper_names, term_names, conf_names, author_flag, paper_flag, term_flag, conf_flag = construct_graph()

    s = 10
    author_idx_traces = {}
    for paper_idx in tqdm.trange(hg.num_nodes("paper")):#hg.num_nodes("conf") = 16
        # print([conf_idx] * num_walks_per_node): [0,0,0,0,0...]
        # print(["cp", "pa", "ap", "pc"] * walk_length): ['cp', 'pa', 'ap', 'pc', 'cp', 'pa', 'ap', 'pc', ...]

        if (paper_flag[paper_idx] == 0): # meaning the node not exist in the graph
            continue
        traces, _ = dgl.sampling.random_walk(
            hg,
            [paper_idx] * num_walks_per_node,
            metapath=["pt", "tp"] * walk_length,
        )
        # author_idx_traces[author_idx] = traces

        s-=1
        m=10
        for tr in traces:
            m-=1
            selected_names = []
            for i in range(0, len(tr)):
                if (i % 2 == 0):
                    selected = paper_names[tr[i]]
                else:
                    selected = term_names[tr[i]]
                selected_names.append(selected)
            outline = " ".join(selected_names)
            print(outline, file=output_path)
        #     if m == 0:
        #         break
        # if s == 0:
        #     break
    # # print(author_idx_traces[0])
    # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
    # # print(author_idx_traces[1])
    # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
    # print(author_idx_traces[10])
    output_path.close()

def generate_metapath_pcp():
    output_path = open("pcp_path.txt", "w")
    count = 0

    hg, author_names, paper_names, term_names, conf_names, author_flag, paper_flag, term_flag, conf_flag = construct_graph()

    s = 10
    author_idx_traces = {}
    for paper_idx in tqdm.trange(hg.num_nodes("paper")):#hg.num_nodes("conf") = 16
        # print([conf_idx] * num_walks_per_node): [0,0,0,0,0...]
        # print(["cp", "pa", "ap", "pc"] * walk_length): ['cp', 'pa', 'ap', 'pc', 'cp', 'pa', 'ap', 'pc', ...]

        if (paper_flag[paper_idx] == 0): # meaning the node not exist in the graph
            continue
        traces, _ = dgl.sampling.random_walk(
            hg,
            [paper_idx] * num_walks_per_node,
            metapath=["pc", "cp"] * walk_length,
        )
        # author_idx_traces[author_idx] = traces

        s-=1
        m=10
        for tr in traces:
            m-=1
            selected_names = []
            for i in range(0, len(tr)):
                if (i % 2 == 0):
                    selected = paper_names[tr[i]]
                else:
                    selected = conf_names[tr[i]]
                selected_names.append(selected)
            outline = " ".join(selected_names)
            print(outline, file=output_path)
        #     if m == 0:
        #         break
        # if s == 0:
        #     break
    # # print(author_idx_traces[0])
    # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
    # # print(author_idx_traces[1])
    # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
    # print(author_idx_traces[10])
    output_path.close()


def generate_metapath_paptp():
    output_path = open("paptp_path.txt", "w")
    count = 0

    hg, author_names, paper_names, term_names, conf_names, author_flag, paper_flag, term_flag, conf_flag = construct_graph()

    s = 10
    author_idx_traces = {}
    for paper_idx in tqdm.trange(hg.num_nodes("paper")):#hg.num_nodes("conf") = 16
        # print([conf_idx] * num_walks_per_node): [0,0,0,0,0...]
        # print(["cp", "pa", "ap", "pc"] * walk_length): ['cp', 'pa', 'ap', 'pc', 'cp', 'pa', 'ap', 'pc', ...]

        if (paper_flag[paper_idx] == 0): # meaning the node not exist in the graph
            continue
        traces, _ = dgl.sampling.random_walk(
            hg,
            [paper_idx] * num_walks_per_node,
            metapath=["pa", "ap", "pt", "tp"] * walk_length,
        )
        # author_idx_traces[author_idx] = traces

        s-=1
        m=10
        for tr in traces:
            m-=1
            selected_names = []
            for i in range(0, len(tr)):
                if (i % 4 == 0):
                    selected = paper_names[tr[i]]
                if (i % 4 == 1):
                    selected = author_names[tr[i]]
                if (i % 4 == 2):
                    selected = paper_names[tr[i]]
                if (i % 4 == 3):
                    selected = term_names[tr[i]]

                selected_names.append(selected)
            outline = " ".join(selected_names)
            print(outline, file=output_path)
        #     if m == 0:
        #         break
        # if s == 0:
        #     break
    # # print(author_idx_traces[0])
    # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
    # # print(author_idx_traces[1])
    # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
    # print(author_idx_traces[10])
    output_path.close()


def generate_metapath_papcp():
    output_path = open("papcp_path.txt", "w")
    count = 0

    hg, author_names, paper_names, term_names, conf_names, author_flag, paper_flag, term_flag, conf_flag = construct_graph()

    s = 10
    author_idx_traces = {}
    for paper_idx in tqdm.trange(hg.num_nodes("paper")):#hg.num_nodes("conf") = 16
        # print([conf_idx] * num_walks_per_node): [0,0,0,0,0...]
        # print(["cp", "pa", "ap", "pc"] * walk_length): ['cp', 'pa', 'ap', 'pc', 'cp', 'pa', 'ap', 'pc', ...]

        if (paper_flag[paper_idx] == 0): # meaning the node not exist in the graph
            continue
        traces, _ = dgl.sampling.random_walk(
            hg,
            [paper_idx] * num_walks_per_node,
            metapath=["pa", "ap", "pc", "cp"] * walk_length,
        )
        # author_idx_traces[author_idx] = traces

        s-=1
        m=10
        for tr in traces:
            m-=1
            selected_names = []
            for i in range(0, len(tr)):
                if (i % 4 == 0):
                    selected = paper_names[tr[i]]
                if (i % 4 == 1):
                    selected = author_names[tr[i]]
                if (i % 4 == 2):
                    selected = paper_names[tr[i]]
                if (i % 4 == 3):
                    selected = conf_names[tr[i]]

                selected_names.append(selected)
            outline = " ".join(selected_names)
            print(outline, file=output_path)
        #     if m == 0:
        #         break
        # if s == 0:
        #     break
    # # print(author_idx_traces[0])
    # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
    # # print(author_idx_traces[1])
    # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
    # print(author_idx_traces[10])
    output_path.close()

if __name__ == "__main__":
    # print("pap_to_edges")
    # generate_metapath_pap()
    # print("ptp_to_edges")
    # generate_metapath_ptp()
    # print("pcp_to_edges")
    # generate_metapath_pcp()
    # print("paptp_to_edges")
    # generate_metapath_paptp()
    # print("papcp_to_edges")
    # generate_metapath_papcp()
    print("aptpa_to_edges")
    generate_metapath_aptpa()
    print("apcpa_to_edges")
    generate_metapath_apcpa()




