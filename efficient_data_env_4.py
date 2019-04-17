from collections import defaultdict
from functools import partial
import time

import numpy as np
import pandas as pd


class Graph(object):
    def __init__(self, train_triples, n_virtual_nodes, n_ents, n_rels):
        """ `train_triples`: all (head, rel, tail) in `train`
            `n_ents`: all real nodes in `train` + `valid` + `test`
            `n_rels`: all real relations in `train` + `valid` + `test`

            Virtual nodes should connected to all real nodes in `train` + `valid` + `test`
            via two types of virtual relation: `into_virtual` and `outof_virtual`
        """
        self.full_train = train_triples
        self.n_full_train = len(self.full_train)

        self.virtual_nodes = [n_ents + i for i in range(n_virtual_nodes)]
        self.n_entities = n_ents + n_virtual_nodes  # including n virtual nodes

        self.into_virtual = n_rels
        self.outof_virtual = n_rels + 1
        self.n_relations = n_rels + 2  # including two virtual relations but not 'selfloop' and 'backtrace'

        self.selfloop = self.n_relations
        self.backtrace = self.n_relations + 1
        self.n_aug_relations = self.n_relations + 2  # including 'selfloop' and 'backtrace'

        full_edges = np.array(train_triples.tolist() + self._add_virtual_edges(self.virtual_nodes, n_ents),
                              dtype='int32').view('<i4,<i4,<i4')  # np.array
        full_edges = np.sort(full_edges, axis=0, order=['f0', 'f2', 'f1']).view('<i4')
        self.n_full_edges = len(full_edges)
        # `full_edges`: including virtual edges but not selfloop edges and backtrace edges
        # full_edges[i] = [id, head, rel, tail] sorted by head, tail, rel with ascending and consecutive `id`s
        self.full_edges = np.concatenate((np.expand_dims(np.arange(self.n_full_edges, dtype='int32'), 1),
                                          full_edges), axis=1)

        self.edge2id = self._make_edge2id(self.full_edges)
        self.count_dct = self._count_over_full_edges(self.full_edges)

        # `edges`: remove the current train triples
        # edges[i] = [id, head, rel, tail] sorted by head, tail, rel with ascending but not consecutive `id`s
        self.edges = None
        self.edges_pd = None
        self.n_edges = 0

        self.past_transitions = None  # past_transitions[i] = [eg_idx, vi, vj, rel, step] (not sorted)
        self.past_trans_attention = None  # past_trans_attention[i] = trans_attention

        self.visited_nodes = None  # visited_nodes[i] = [eg_idx, v] sorted by ed_idx, v

        self.node_attention_li = None
        self.attend_nodes_li = None

    def _add_virtual_edges(self, virtual_nodes, n_ents):
        """ `n_ents`: all real nodes in `train` + `valid` + `test`
        """
        virtual_edges = []
        for i in range(n_ents):
            for vir in virtual_nodes:
                virtual_edges.append((i, self.into_virtual, vir))
                virtual_edges.append((vir, self.outof_virtual, i))
        return virtual_edges

    def _count_over_full_edges(self, edges):
        dct = defaultdict(int)
        for i, h, r, t in edges:
            dct[h] += 1
            dct[(h, r)] += 1
            dct[(h, r, t)] += 1
        return dct

    def _make_edge2id(self, edges):
        dct = defaultdict(set)
        for i, h, r, t in edges:
            dct[(h, r, t)].add(i)
        return dct

    def count(self, t):
        return self.count_dct[t]

    def draw_train(self, n_train):
        shuffled_idx = np.random.permutation(self.n_full_train)
        start = 0
        while start < self.n_full_train:
            end = min(start + n_train, self.n_full_train)
            train = self.full_train[shuffled_idx[start:end]]
            train_eids = set()
            for h, r, t in train:
                train_eids.update(self.edge2id[(h, r, t)])
            graph_eids = set()
            for id_set in self.edge2id.values():
                graph_eids.update(id_set)
            graph_eids = graph_eids - train_eids
            graph_eids = np.sort(np.array(list(graph_eids), dtype='int32'))
            self.edges = self.full_edges[graph_eids]
            self.edges_pd = pd.DataFrame({'edge_id': self.edges[:, 0], 'vi': self.edges[:, 1],
                                          'rel': self.edges[:, 2], 'vj': self.edges[:, 3]})
            self.n_edges = len(self.edges)
            yield train, self
            start = end

    def get_graph_edges(self):
        return np.stack([np.zeros((self.n_edges,), dtype='int32'), self.edges[:, 0]], axis=1)  # (0, edge_id)

    def use_full_edges(self):
        self.edges = self.full_edges
        self.edges_pd = pd.DataFrame({'edge_id': self.edges[:, 0], 'vi': self.edges[:, 1],
                                      'rel': self.edges[:, 2], 'vj': self.edges[:, 3]})
        self.n_edges = len(self.edges)

    def get_selected_edges(self, sampled_edges, tc=None):
        """ `sampled_edges`: n_sampled_edges x 2 ( int32, outputs[i] = (eg_idx, edge_id) )
        """
        if tc is not None:
            t0 = time.time()

        sorted_idx = np.squeeze(np.argsort(sampled_edges.view('<i4,<i4'), order=['f0', 'f1'], axis=0), 1)
        sorted_idx = np.array(sorted_idx, dtype='int32')
        sampled_edges = sampled_edges[sorted_idx]
        selected_edges = self.full_edges[sampled_edges[:, 1]][:, [1,3,2]]  # selected_edges[i] = [head, tail, rel]
        selected_edges = np.concatenate((np.expand_dims(sampled_edges[:, 0], 1), selected_edges), axis=1)  # selected_edges[i] = [eg_idx, head, tail, rel]
        _, idx_vi = np.unique(selected_edges[:, [0, 1]], axis=0, return_inverse=True)
        _, idx_vj = np.unique(selected_edges[:, [0, 2]], axis=0, return_inverse=True)
        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)
        # selected_edges[i] = [eg_idx, head, tail, rel, idx_vi, idx_vj], sorted by (eg_idx, head, tail))
        selected_edges = np.concatenate((selected_edges, idx_vi, idx_vj), axis=1)

        if tc is not None:
            tc['sel_e'] += time.time() - t0
        return selected_edges, sorted_idx

    def get_attended_nodes(self, node_attention, max_nodes, tc=None):
        """ node_attention: batch_size x n_nodes
        """
        if tc is not None:
            t0 = time.time()

        node_attention = node_attention.numpy()
        eps = 1e-20
        eps_noise = np.random.uniform(low=0., high=eps, size=node_attention.shape)
        node_attention = node_attention + eps_noise

        batch_size = node_attention.shape[0]
        n_nodes = node_attention.shape[1]
        max_nodes = min(n_nodes, max_nodes)
        sorted_idx = np.argsort(-node_attention, axis=1)[:, :max_nodes]
        sorted_idx = np.array(sorted_idx, dtype='int32')
        node_attention = np.take_along_axis(node_attention, sorted_idx, axis=1)  # sorted node attention
        mask = (node_attention > 2 * eps)[:, :max_nodes]
        eg_idx = np.repeat(np.expand_dims(np.array(np.arange(batch_size), dtype='int32'), 1), max_nodes, axis=1)
        eg_idx = eg_idx[mask]
        vi = sorted_idx[mask]
        attended_nodes = np.stack([eg_idx, vi], axis=1)

        if tc is not None:
            tc['att_v'] += time.time() - t0
        return attended_nodes  # n_attended_nodes x 2 ( attended_nodes[i] = (eg_idx, vi) )

    # todo: candidate_edges (sorted by (eg_idx, vi))
    def get_candidate_edges(self, attended_nodes, tc=None):
        """ attended_nodes: n_attended_nodes x 2 ( attended_nodes[i] = (eg_idx, vi) )
        """
        if tc is not None:
            t0 = time.time()

        attended_nodes_pd = pd.DataFrame({'eg_idx': attended_nodes[:, 0], 'vi': attended_nodes[:, 1]})
        merged_pd = pd.merge(attended_nodes_pd, self.edges_pd['edge_id', 'vi'], on='vi')
        candidate_edges = merged_pd[['eg_idx', 'edge_id']].sort_values(['eg_idx', 'edge_id']).values  # dtype=int32

        if tc is not None:
            tc['att_e'] += time.time() - t0
        return candidate_edges  # n_candidate_edges x 2 ( candidate_edges[i] = (eg_idx, edge_id), sorted by eg_idx, edge_id )

    def reset_past_transitions(self):
        self.past_transitions = None  # past_transitions[i] = [eg_idx, vi, vj, rel, step]
        self.past_trans_attention = None  # past_trans_attention[i] = trans_attention

    def reset_node_attention_li(self, node_attention):
        self.node_attention_li = [node_attention.numpy()]

    def reset_attended_nodes_li(self):
        self.attend_nodes_li = []

    def get_selfloop_and_backtrace(self, attended_nodes, attended_node_attention,
                                   max_backtrace_edges, step, tc=None, backtrace_decay=1.):
        """ attended_nodes: n_attended_nodes x 2 ( int32, attended_nodes[i] = (eg_idx, vi) ) )
            attended_node_attention: n_attended_nodes
            step: the current step
        """
        if tc is not None:
            t0 = time.time()

        eg_idx, vi = attended_nodes[:, 0], attended_nodes[:, 1]
        selfloop_edges = np.stack([eg_idx, vi, vi, np.repeat(np.array(self.selfloop, dtype='int32'), eg_idx.shape[0])],
                                  axis=1)  # (eg_idx, vi, vi, selfloop)

        if self.past_transitions is None:
            return selfloop_edges, None

        eps = 1e-20
        attended_node_attention = attended_node_attention.numpy()  # n_attended_nodes
        batch_size = np.amax(attended_nodes[:, 0]) + 1
        backtrace_edges = []
        for eg_idx in range(batch_size):
            cur_attended_mask = attended_nodes[:, 0] == eg_idx
            cur_attended = attended_nodes[cur_attended_mask]  # (cur_eg_idx, vi as vj for past_transitions)
            cur_attended_att = attended_node_attention[cur_attended_mask]  # vi_att as vj_att

            cur_trans_mask = self.past_transitions[:, 0] == eg_idx
            cur_trans = self.past_transitions[cur_trans_mask]  # (cur_eg_idx, vi, vj, rel, step)
            cur_trans_att = self.past_trans_attention[cur_trans_mask]  # trans_att

            cur_trans_pd = pd.DataFrame({'eg_idx': cur_trans[:, 0],
                                         'vi': cur_trans[:, 1],
                                         'vj': cur_trans[:, 2],
                                         'rel': cur_trans[:, 3],
                                         'step': cur_trans[:, 4],
                                         'trans_att': cur_trans_att + np.random.uniform(low=0.,
                                                                                        high=eps,
                                                                                        size=cur_trans_att.shape)})
            cur_attended_pd = pd.DataFrame({'vj': cur_attended[:, 1],
                                            'vj_att': cur_attended_att})
            result = pd.merge(cur_trans_pd, cur_attended_pd, on='vj')
            result['att'] = result['trans_att'] * result['vj_att'] * (backtrace_decay ** (step - result['step']))
            max_backtrace_edges = min(max_backtrace_edges, result.shape[0])
            result = result.nlargest(max_backtrace_edges, 'att')
            result = result[['eg_idx', 'vj', 'vi']].sort_values(['vj', 'vi'])
            backtrace_edges.append(np.concatenate([np.array(result.values, dtype='int32'),
                                                   np.expand_dims(np.repeat(np.array(self.backtrace, dtype='int32'),
                                                                            result.shape[0]),
                                                                  1)], axis=1))  # (cur_eg_idx, vj, vi, backtrace)
        backtrace_edges = np.concatenate(backtrace_edges, axis=0)

        if tc is not None:
            tc['sl_bt'] += time.time() - t0
        return selfloop_edges, backtrace_edges  # (eg_idx, vi, vi, selfloop), (eg_idx, vj, vi, backtrace)

    def get_union_edges(self, selected_edges, selfloope_edges, backtrace_edges, tc=None):
        """ selected_edges: n_selected_edges x 6 (eg_idx, vi, vj, rel, idx_vi, idx_vj)
            selfloop_edges: n_selfloop_edges x 4 (eg_idx, vi, vi, selfloop)
            backtrace_edges: n_backtrace_edges x 4 (eg_idx, vj, vi, backtrace)
        """
        if tc is not None:
            t0 = time.time()

        selected_edges = selected_edges[:, :4]  # (eg_idx, vi, vj, rel)
        n_selected_edges = selected_edges.shape[0]
        if backtrace_edges is None:
            all_edges = np.concatenate([selected_edges, selfloope_edges], axis=0)
        else:
            all_edges = np.concatenate([selected_edges, selfloope_edges, backtrace_edges], axis=0)
        sorted_idx = np.squeeze(np.argsort(all_edges.view('<i4,<i4,<i4,<i4'), order=['f0', 'f1', 'f2'], axis=0), 1)
        sorted_idx = np.array(sorted_idx, dtype='int32')

        new_idx = np.array(np.argsort(sorted_idx), dtype='int32')
        new_idx_for_edges_y = np.expand_dims(new_idx[:n_selected_edges], 1)
        rest_idx = np.expand_dims(new_idx[n_selected_edges:], 1)

        selected_edges = all_edges[sorted_idx]
        _, idx_vi = np.unique(selected_edges[:, [0, 1]], axis=0, return_inverse=True)
        _, idx_vj = np.unique(selected_edges[:, [0, 2]], axis=0, return_inverse=True)
        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)
        selected_edges = np.concatenate((selected_edges, idx_vi, idx_vj), axis=1)

        if tc is not None:
            tc['union_e'] += time.time() - t0
        # selected_edges: n_selected_edges x 6 (eg_idx, vi, vj, rel, idx_vi, idx_vj)
        return selected_edges, new_idx_for_edges_y, rest_idx

    def store_transitions(self, selected_edges, selected_trans_att, step, tc=None):
        """ selected_edges: n_selected_edges x 6 (eg_idx, vi, vj, rel, idx_vi, idx_vj)
                (not including selfloope_edges and backtrace_edges)
            selected_trans_att: n_selected_edges
        """
        if tc is not None:
            t0 = time.time()

        transitions = np.concatenate([selected_edges[:, :4],
                                      np.expand_dims(
                                          np.repeat(np.array(step, dtype='int32'), selected_edges.shape[0]), 1)],
                                     axis=1)  # n_selected_edges x 4 (eg_idx, vi, vj, rel, step)
        trans_attention = selected_trans_att.numpy()  # n_selected_edges

        if self.past_transitions is None:
            self.past_transitions = transitions # past_transitions[i] = [eg_idx, vi, vj, rel, step]
            self.past_trans_attention = trans_attention # past_trans_attention[i] = trans_attention
        else:
            self.past_transitions = np.concatenate([self.past_transitions, transitions], axis=0)
            self.past_trans_attention = np.concatenate([self.past_trans_attention, trans_attention], axis=0)

        if tc is not None:
            tc['st_trans'] += time.time() - t0

    def store_node_attention(self, node_attention):
        self.node_attention_li.append(node_attention.numpy())

    def store_attended_nodes(self, attended_nodes):
        self.attend_nodes_li.append(attended_nodes)

    def get_initial_selected_nodes(self, heads, tc=None):
        """ heads: batch_size
        """
        if tc is not None:
            t0 = time.time()

        batch_size = heads.shape[0]
        selected_v = np.stack([np.array(np.arange(batch_size), dtype='int32'), heads], axis=1)  # n_selected_nodes (=batch_size) x 2

        if tc is not None:
            tc['i_sel_v'] += time.time() - t0
        return selected_v

    def reset_visited_nodes(self, selected_v):
        """ selected_v: n_selected_nodes (=batch_size) x 2, (idx, v)
        """
        self.visited_nodes = selected_v  # n_visited_nodes x 2, (eg_idx, v) sorted by ed_idx, v

    def update_visited_nodes(self, selected_edges, tc=None):
        """ selected_edges: n_selected_edges (=n_attended_edges) x 6, (idx, vi, vj, rel, idx_vi, idx_vj) sorted by (idx, vi, vj)
        """
        if tc is not None:
            t0 = time.time()

        selected_vj = np.unique(selected_edges[:, [0, 2]], axis=0)
        mask = np.in1d(selected_vj.view('<i4,<i4'), self.visited_nodes.view('<i4,<i4'), assume_unique=True)
        mask = np.logical_not(mask)
        new_nodes = selected_vj[mask]  # n_new_nodes x 2
        prev_visited_nodes = self.visited_nodes
        visited_nodes = np.concatenate([prev_visited_nodes, new_nodes], axis=0)  # n_visited_nodes (new) x 2
        sorted_idx = np.squeeze(np.argsort(visited_nodes.view('<i4,<i4'), order=['f0', 'f1'], axis=0), 1)
        sorted_idx = np.array(sorted_idx, dtype='int32')

        self.visited_nodes = visited_nodes[sorted_idx]
        n_visited_nodes = len(self.visited_nodes)

        new_idx = np.array(np.argsort(sorted_idx), dtype='int32')
        n_previous_nodes = prev_visited_nodes.shape[0]
        new_idx_for_previous = np.expand_dims(new_idx[:n_previous_nodes], 1)
        rest_idx = np.expand_dims(new_idx[n_previous_nodes:], 1)

        if tc is not None:
            tc['up_vis'] += time.time() - t0
        return new_idx_for_previous, rest_idx, n_visited_nodes

    def index_visited_nodes(self, selected_edges, tc=None):
        """ selected_edges (aug_selected_edges): n_selected_edges (=n_aug_selected_edges) x 6
        """
        if tc is not None:
            t0 = time.time()

        selected_vi = np.unique(selected_edges[:, [0, 1]], axis=0)
        selected_vj = np.unique(selected_edges[:, [0, 2]], axis=0)
        mask_vi = np.in1d(self.visited_nodes.view('<i4,<i4'), selected_vi.view('<i4,<i4'), assume_unique=True)
        mask_vj = np.in1d(self.visited_nodes.view('<i4,<i4'), selected_vj.view('<i4,<i4'), assume_unique=True)
        new_idx_e2vi = np.array(np.argwhere(mask_vi), dtype='int32')  # n_selected_by_idx_vi x 1
        new_idx_e2vj = np.array(np.argwhere(mask_vj), dtype='int32')  # n_selected_by_idx_vj x 1

        idx_vi = selected_edges[:, 4]
        idx_vj = selected_edges[:, 5]

        try:
            new_idx_e2vi = new_idx_e2vi[idx_vi]  # n_selected_edges x 1
            new_idx_e2vj = new_idx_e2vj[idx_vj]  # n_selected_edges x 1
        except IndexError:
            print(len(new_idx_e2vi))

        # selected_edges:n_selected_edges x 8, (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by idx, vi, vj
        selected_edges = np.concatenate([selected_edges, new_idx_e2vi, new_idx_e2vj], axis=1)

        if tc is not None:
            tc['idx_vis'] += time.time() - t0
        return selected_edges

    # todo: changed here
    def sample_edges(self, candidate_edges, edges_logits, max_edges_per_node, tc=None):
        """ candidate_edges: (pd.DataFram) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel)
            edges_logits: (Tensor) n_full_edges
        """
        if tc is not None:
            t0 = time.time()

        edges_logits = edges_logits.numpy()
        edge_id = candidate_edges['edge_id'].values
        logits = edges_logits[edge_id]  # n_candidate_edges

        eps = 1e-20
        loglog_u = - np.log(- np.log(np.random.uniform(size=(len(candidate_edges),)) + eps) + eps)
        loglog_u = np.array(loglog_u, dtype='float32')  # n_candidate_edges
        logits = logits + loglog_u  # n_candidate_edges

        candidate_edges.assign(logits=logits)
        # sampled_edges: (pd.DataFrame) n_sampled_edges x 4, (eg_idx, vi, level_2, logits)
        sampled_edges = candidate_edges.groupby(['eg_idx', 'vi'])['logits'].nlargest(max_edges_per_node).reset_index()

        # sampled_edges: (pd.DataFrame) n_sampled_edges x 7, (eg_idx, vi, level_2, logits, vj, rel, edge_id)
        sampled_edges = pd.merge(sampled_edges, candidate_edges[['vj', 'rel', 'edge_id']],
                                 left_on='level_2', right_index=True)

        # loglog_u: n_candidate_edges
        # sampled_edges: (pd.DataFrame) n_sampled_edges x 7, (eg_idx, vi, level_2, logits, vj, rel, edge_id)
        return loglog_u, sampled_edges


class DataFeeder(object):
    def get_train_batch(self, train_data, batch_size, shuffle=True):
        n_train = len(train_data)
        rand_idx = np.random.permutation(n_train) if shuffle else np.arange(n_train)
        start = 0
        while start < n_train:
            end = min(start + batch_size, n_train)
            batch = [train_data[i] for i in rand_idx[start:end]]
            yield np.array(batch, dtype='int32'), end - start
            start = end

    def get_eval_batch(self, eval_data, batch_size):
        n_eval = len(eval_data)
        start = 0
        while start < n_eval:
            end = min(start + batch_size, n_eval)
            batch = eval_data[start:end]
            yield np.array(batch, dtype='int32'), end - start
            start = end


class DataEnv(object):
    def __init__(self, dataset, hparams):
        self.hparams = hparams
        self.data_feeder = DataFeeder()

        self.valid_triples = dataset.valid
        self.test_triples = dataset.test
        self.n_valid = len(self.valid_triples)
        self.n_test = len(self.test_triples)

        # `graph`:
        # (1) graph.entities: train + valid + test + { n virtual nodes }
        # (2) graph.relations: train + valid + test + { 'into_virtual', 'outof_virtual' }
        # (3) graph.full_edges: train + virtual edges
        # (4) graph.edges: train - split_train + virtual edges
        self.graph = Graph(dataset.train, hparams.n_virtual_nodes, dataset.n_entities, dataset.n_relations)

        self.filter_pool = defaultdict(set)
        for head, rel, tail in np.concatenate([self.graph.full_train, self.valid_triples, self.test_triples], axis=0):
            self.filter_pool[(head, rel)].add(tail)

    def draw_train(self, n_train):
        for train, graph in self.graph.draw_train(n_train):
            yield self.get_train_batcher(train), graph

    def get_train_batcher(self, train):
        return partial(self.data_feeder.get_train_batch, train, shuffle=False)

    def get_valid_batcher(self):
        return partial(self.data_feeder.get_eval_batch, self.valid_triples)

    def get_test_batcher(self):
        return partial(self.data_feeder.get_eval_batch, self.test_triples)
