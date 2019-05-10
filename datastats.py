from collections import defaultdict, Counter
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import datasets

class DataStats(object):
    def __init__(self):
        pass

    def shortest_path_length_report(self, train_triples, valid_triples, test_triples):
        """ train_triples (train_eg_triples): (np.array) n_edges x 3, (vi, vj, rel)
            valid_triples: (np.array) n_edges x 3, (vi, vj, rel)
            test_triples: (np.array) n_edges x 3, (vi, vj, rel)
        """
        full_edges = train_triples
        graph = nx.MultiDiGraph(full_edges[:, :2].tolist())

        n_pairs_with_unseen_nodes = 0
        n_pairs_with_no_paths = 0
        max_length = -1
        min_length = -1
        avg_length = 0
        count = 0
        for vi, vj, rel in valid_triples:
            try:
                length = nx.algorithms.shortest_path_length(graph, source=vi, target=vj)
                max_length = max(length, max_length) if max_length != -1 else length
                min_length = min(length, min_length) if min_length != -1 else length
                avg_length += length
                count += 1
            except nx.NodeNotFound:
                n_pairs_with_unseen_nodes += 1
            except nx.NetworkXNoPath:
                n_pairs_with_no_paths += 1
        avg_length = avg_length / count
        print('--- VALID ---')
        print('avg_length: {}'.format(avg_length))
        print('max_length: {}'.format(max_length))
        print('min_length: {}'.format(min_length))
        print('n_paris_with_paths: {}'.format(count))
        print('n_pairs_with_unseen_nodes: {}'.format(n_pairs_with_unseen_nodes))
        print('n_pairs_with_no_paths: {}'.format(n_pairs_with_no_paths))

        n_pairs_with_unseen_nodes = 0
        n_pairs_with_no_paths = 0
        max_length = -1
        min_length = -1
        avg_length = 0
        count = 0
        for vi, vj, rel in test_triples:
            try:
                length = nx.algorithms.shortest_path_length(graph, source=vi, target=vj)
                max_length = max(length, max_length) if max_length != -1 else length
                min_length = min(length, min_length) if min_length != -1 else length
                avg_length += length
                count += 1
            except nx.NodeNotFound:
                n_pairs_with_unseen_nodes += 1
            except nx.NetworkXNoPath:
                n_pairs_with_no_paths += 1
        avg_length = avg_length / count
        print('--- TEST ---')
        print('avg_length: {}'.format(avg_length))
        print('max_length: {}'.format(max_length))
        print('min_length: {}'.format(min_length))
        print('n_paris_with_paths: {}'.format(count))
        print('n_pairs_with_unseen_nodes: {}'.format(n_pairs_with_unseen_nodes))
        print('n_pairs_with_no_paths: {}'.format(n_pairs_with_no_paths))

        """
        full_train = train_triples
        n_full_train = len(train_triples)
        n_train = len(test_triples)
        shuffled_idx = np.random.permutation(n_full_train)
        start = 0
        graph_i = 0
        while start < n_full_train:
            end = min(start + n_train, n_full_train)
            train = full_train[shuffled_idx[start:end]]
            train_e = set()
            for h, t, r in train:
                train_e.add((h, t, r))
            graph_e = set()
            for h, t, r in full_train:
                if (h, t, r) not in train_e:
                    graph_e.add((h, t, r))

            graph = nx.MultiDiGraph([(vi, vj) for vi, vj, rel in graph_e])

            n_pairs_with_unseen_nodes = 0
            n_pairs_with_no_paths = 0
            max_length = -1
            min_length = -1
            avg_length = 0
            count = 0
            for vi, vj, rel in valid_triples:
                try:
                    length = nx.algorithms.shortest_path_length(graph, source=vi, target=vj)
                    max_length = max(length, max_length) if max_length != -1 else length
                    min_length = min(length, min_length) if min_length != -1 else length
                    avg_length += length
                    count += 1
                except nx.NodeNotFound:
                    n_pairs_with_unseen_nodes += 1
                except nx.NetworkXNoPath:
                    n_pairs_with_no_paths += 1
            avg_length = avg_length / count
            print('--- TRAIN {} ---'.format(graph_i))
            print('avg_length: {}'.format(avg_length))
            print('max_length: {}'.format(max_length))
            print('min_length: {}'.format(min_length))
            print('n_paris_with_paths: {}'.format(count))
            print('n_pairs_with_unseen_nodes: {}'.format(n_pairs_with_unseen_nodes))
            print('n_pairs_with_no_paths: {}'.format(n_pairs_with_no_paths))
            graph_i += 1

            start = end
        """

    def avg_shortest_path_length_report(self, train_triples):
        graph = nx.MultiDiGraph(train_triples[:, :2].tolist())
        n_pairs_with_unseen_nodes = 0
        n_pairs_with_no_paths = 0
        nodes = list(graph.nodes)
        n_nodes = len(nodes)
        avg_length = 0
        max_length = -1
        min_length = -1
        N = 100000
        for _ in range(N):
            src = nodes[random.randint(0, n_nodes - 1)]
            dst = nodes[random.randint(0, n_nodes - 1)]
            try:
                length = nx.algorithms.shortest_path_length(graph, source=src, target=dst)
                avg_length += length
                max_length = max(max_length, length) if max_length != -1 else length
                min_length = min(min_length, length) if min_length != -1 else length
                # print(src, dst, length)
            except nx.NodeNotFound:
                n_pairs_with_unseen_nodes += 1
            except nx.NetworkXNoPath:
                n_pairs_with_no_paths += 1
        print('n_pairs_with_unseen_nodes: {}'.format(n_pairs_with_unseen_nodes))
        print('n_pairs_with_no_paths: {}'.format(n_pairs_with_no_paths))
        print('avg_shortest_length: {}'.format(avg_length / N))
        print('min_shortest_length: {}'.format(min_length))
        print('max_shortest_length: {}'.format(max_length))

    def degree_histogram_report(self, train_triples):
        graph = nx.Graph(train_triples[:, :2].tolist())
        degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence

        print('n_nodes: {}'.format(len(graph.nodes)))
        print('n_edges: {}'.format(len(graph.edges)))
        print('n_greater_than_1000: {}'.format(len([d for d in degree_sequence if d > 1000])))
        print('n_greater_than_100: {}'.format(len([d for d in degree_sequence if d > 100])))
        print('max_degree: {}'.format(degree_sequence[0]))
        print('min_degree: {}'.format(degree_sequence[-1]))
        print('avg_degree: {}'.format(np.sum(degree_sequence) / len(degree_sequence)))

        # print "Degree sequence", degree_sequence
        degreeCount = Counter(degree_sequence)
        deg, cnt = zip(*filter(lambda x: x[0] < 50, degreeCount.items()))

        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color='b')

        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)

        # draw graph in inset
        plt.axes([0.4, 0.4, 0.5, 0.5])
        #Gcc = sorted(nx.connected_component_subgraphs(graph), key=len, reverse=True)[0]
        #pos = nx.spring_layout(graph)
        plt.axis('off')
        #nx.draw_networkx_nodes(graph, pos, node_size=20)
        #nx.draw_networkx_edges(graph, pos, alpha=0.4)

        plt.show()

    def common_report(self, train_path, valid_path, test_path):
        ent_train = set()
        rel_train = set()
        n_rec = 0
        with open(train_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                ent_train.add(h)
                rel_train.add(r)
                ent_train.add(t)
                n_rec = n_rec + 1
        print('#entities in train: {}'.format(len(ent_train)))
        print('#relations in train: {}'.format(len(rel_train)))
        print('#records in train: {}'.format(n_rec))
        print()

        ent_valid = set()
        rel_valid = set()
        n_rec = 0
        with open(valid_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                ent_valid.add(h)
                rel_valid.add(r)
                ent_valid.add(t)
                n_rec = n_rec + 1
        print('#entities in valid: {}'.format(len(ent_valid)))
        print('#relations in valid: {}'.format(len(rel_valid)))
        print('#records in train: {}'.format(n_rec))
        print()

        ent_test = set()
        rel_test = set()
        n_rec = 0
        with open(test_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                ent_test.add(h)
                rel_test.add(r)
                ent_test.add(t)
                n_rec = n_rec + 1
        print('#entities in test: {}'.format(len(ent_test)))
        print('#relations in test: {}'.format(len(rel_test)))
        print('#records in train: {}'.format(n_rec))
        print()

        print('#entities in valid but not in train: {}'.format(len(ent_valid - ent_train)))
        print('#relations in valid but not in train: {}'.format(len(rel_valid - rel_train)))
        print()

        print('#entities in test but not in train: {}'.format(len(ent_test - ent_train)))
        print('#relations in test but not in train: {}'.format(len(rel_test - rel_train)))
        print()

        deg_train = defaultdict(lambda: np.zeros(3, dtype=np.int))
        with open(train_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                deg_train[h] = deg_train[h] + np.array([1, 0, 1])
                deg_train[t] = deg_train[t] + np.array([0, 1, 1])

        deg_train_np = np.stack(list(deg_train.values()))
        print('max head degree in train: {} where head is {}'.format(deg_train_np[:, 0].max(),
                                                                     list(deg_train.keys())[deg_train_np[:, 0].argmax()]))
        print('max tail degree in train: {} where tail is {}'.format(deg_train_np[:, 1].max(),
                                                                     list(deg_train.keys())[deg_train_np[:, 1].argmax()]))
        print('max degree in train: {}'.format(deg_train_np[:, 2].max()))
        print()

    def dataset_properties(self, graph_triples, eg_triples, is_train=True):
        ht_dct = defaultdict(list)
        htr_dct = defaultdict(int)
        r_dct = defaultdict(int)
        for h, t, r in graph_triples:
            ht_dct[(h, t)].append(r)
            htr_dct[(h, t, r)] += 1
            r_dct[r] += 1

        n_same_ht = 0
        n_same_htr = 0
        n_inverse_th = 0
        n_inverse_thr = 0
        n_total = 0
        for h, t, r in eg_triples:
            if (is_train and len(ht_dct[(h, t)]) > 1) or (not is_train and len(ht_dct[(h, t)]) > 0):
                n_same_ht += 1
            if (is_train and htr_dct[(h, t, r)] > 1) or (not is_train and htr_dct[(h, t, r)] > 0):
                n_same_htr += 1
            if len(ht_dct[(t, h)]) > 0:
                n_inverse_th += 1
            if htr_dct[(t, h, r)] > 0:
                n_inverse_thr += 1
            n_total += 1
        print('n_same_ht: {:d} | n_same_htr: {:d} | n_inverse_th: {:d} | n_inverse_thr: {:d} | n_total: {:d}'.format(
            n_same_ht, n_same_htr, n_inverse_th, n_inverse_thr, n_total))

        r_set = set([r for _, _, r in eg_triples])
        n_eg_with_r = 0
        for r in r_set:
            n_eg_with_r += r_dct[r]
        print('n_rel_ratio: {:f} | n_eg_ratio: {:f} | n_eg_with_r_ratio: {:f}'.format(
            len(r_set)/len(r_dct), len(eg_triples)/len(graph_triples), n_eg_with_r/len(graph_triples)))

        full_edges = set([(h, t) for h, t, r in graph_triples])
        n_pairs_with_unseen_nodes = 0
        n_pairs_with_no_paths = 0
        max_length = -1
        min_length = -1
        avg_length = 0
        count = 0

        graph = nx.Graph(list(full_edges))
        for h, t, r in eg_triples:
            if is_train:
                if h <= t:
                    graph.remove_edge(h, t)
                else:
                    graph.remove_edge(t, h)
            try:
                length = nx.algorithms.shortest_path_length(graph, source=h, target=t)
                max_length = max(length, max_length) if max_length != -1 else length
                min_length = min(length, min_length) if min_length != -1 else length
                avg_length += length
                count += 1
            except nx.NodeNotFound:
                n_pairs_with_unseen_nodes += 1
            except nx.NetworkXNoPath:
                n_pairs_with_no_paths += 1
            if is_train:
                if h <= t:
                    graph.add_edge(h, t)
                else:
                    graph.add_edge(h, t)

        avg_length = avg_length / count
        print('avg_length: {}'.format(avg_length))
        print('max_length: {}'.format(max_length))
        print('min_length: {}'.format(min_length))
        print('n_paris_with_paths: {}'.format(count))
        print('n_pairs_with_unseen_nodes: {}'.format(n_pairs_with_unseen_nodes))
        print('n_pairs_with_no_paths: {}'.format(n_pairs_with_no_paths))

        avg_degree_h = 0
        avg_degree_t = 0
        for h, t, r in eg_triples:
            if h in graph.nodes:
                avg_degree_h += graph.degree[h]
            if t in graph.nodes:
                avg_degree_t += graph.degree[t]
        avg_degree_h /= len(eg_triples)
        avg_degree_t /= len(eg_triples)
        print('avg_degree_h: {:f} | avg_degree_t: {:f}'.format(avg_degree_h, avg_degree_t))


def report_1(ds):
    stats = DataStats()
    stats.shortest_path_length_report(ds.train, ds.valid, ds.test)


def report_2(ds):
    stats = DataStats()
    stats.avg_shortest_path_length_report(ds.train)


def report_3(ds):
    stats = DataStats()
    stats.degree_histogram_report(ds.train)


def report_4(ds):
    graph = nx.Graph(ds.graph[:, :2].tolist())
    pos = nx.spring_layout(graph)
    node_color = [float(graph.degree(v)) for v in graph]
    edge_color = [node_color[vi] + node_color[vj] for vi, vj in graph.edges]

    plt.figure(figsize=(24, 32))

    nx.draw_networkx_nodes(graph, pos, node_size=200, node_color=node_color, alpha=0.4)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color=edge_color)

    labels = {v: ds.id2entity[v] for v in graph}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)

    plt.axis('off')
    plt.savefig('graph_1.pdf')
    plt.show()

    plt.figure(figsize=(24, 32))

    nx.draw_networkx_nodes(graph, pos, node_size=200, node_color=node_color, alpha=0.4)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color=edge_color)

    nx.draw_networkx_labels(graph, pos, font_size=8)

    plt.axis('off')
    plt.savefig('graph_2.pdf')
    plt.show()

def report_5(ds):
    graph = nx.DiGraph(ds.graph[:, :2].tolist())
    pos = nx.spring_layout(graph)
    node_color = [float(graph.degree(v)) for v in graph]
    edge_color = [node_color[vi] + node_color[vj] for vi, vj in graph.edges]

    plt.figure(figsize=(24, 32))

    nx.draw_networkx_nodes(graph, pos, node_size=80, node_color=node_color, alpha=0.3)
    nx.draw_networkx_edges(graph, pos, alpha=0.1, edge_color=edge_color, arrowstyle='->', arrowsize=3)

    labels = {v: ds.id2entity[v] for v in graph}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=4)

    plt.axis('off')
    plt.savefig('graph_1.pdf')
    plt.show()

    plt.figure(figsize=(24, 32))

    nx.draw_networkx_nodes(graph, pos, node_size=80, node_color=node_color, alpha=0.3)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color=edge_color, arrowstyle='->', arrowsize=5)

    nx.draw_networkx_labels(graph, pos, font_size=4)

    plt.axis('off')
    plt.savefig('graph_2.pdf')
    plt.show()

def report_6(ds):
    stats = DataStats()
    print('\ntrain set:')
    stats.dataset_properties(ds.graph, ds.train, is_train=True)
    print('\nvalid set:')
    stats.dataset_properties(ds.graph, ds.valid, is_train=False)
    print('\ntest set:')
    stats.dataset_properties(ds.graph, ds.test, is_train=False)

if __name__ == '__main__':
    for ds in datasets.Nell995.datasets():
        print(ds.name)
        report_6(ds)
