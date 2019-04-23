from collections import defaultdict, namedtuple
from functools import partial
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import get_segment_ids, get_unique, groupby_2cols_nlargest, groupby_1cols_nlargest, groupby_1cols_merge


VirtualRelations = namedtuple('VirtualRelations',
                              ['real2virtual', 'virtual2real', 'virtual2virtual',
                               'virtual2vcenter', 'vcenter2virtual', 'vcenter2vcenter'])

class Graph(object):
    def __init__(self, train_triples, n_ents, n_rels, hparams):
        """ `train_triples`: all (head, tail, rel) in `train`
            `n_ents`: all real nodes in `train` + `valid` + `test`
            `n_rels`: all real relations in `train` + `valid` + `test`

            Virtual nodes should connected to all real nodes in `train` + `valid` + `test`
            via two types of virtual relation: `into_virtual` and `outof_virtual`
        """
        self.full_train = train_triples
        self.n_full_train = len(self.full_train)

        self.virtual_relations = VirtualRelations(n_rels, n_rels + 1, n_rels + 2, n_rels + 3, n_rels + 4, n_rels + 5)
        self.virtual_nodes, self.virtual_edges = self._add_virtual_edges(hparams.n_clusters_per_clustering,
                                                                         hparams.n_clustering,
                                                                         hparams.connected_clustering,
                                                                         n_ents,
                                                                         n_rels)
        self.n_entities = n_ents + len(self.virtual_nodes)  # including n virtual nodes
        self.n_relations = n_rels + len(self.virtual_relations)  # including virtual relations but not 'selfloop' and 'backtrace'

        full_edges = np.array(train_triples.tolist() + self.virtual_edges, dtype='int32').view('<i4,<i4,<i4')
        full_edges = np.sort(full_edges, axis=0, order=['f0', 'f1', 'f2']).view('<i4')
        self.n_full_edges = len(full_edges)
        # `full_edges`: including virtual edges but not selfloop edges and backtrace edges
        # full_edges[i] = [id, head, tail, rel] sorted by head, tail, rel with ascending and consecutive `id`s
        self.full_edges = np.concatenate([np.expand_dims(np.arange(self.n_full_edges, dtype='int32'), 1),
                                          full_edges], axis=1)

        self.selfloop = self.n_relations
        self.backtrace = self.n_relations + 1
        self.n_aug_relations = self.n_relations + 2  # including 'selfloop' and 'backtrace'

        self.edge2id = self._make_edge2id(self.full_edges)
        self.count_dct = self._count_over_full_edges(self.full_edges)

        # `edges`: remove the current train triples
        # edges[i] = [id, head, tail, rel] sorted by head, tail, rel with ascending but not consecutive `id`s
        self.edges = None
        self.n_edges = 0

        self.past_transitions = None  # (pd.DataFrame) (eg_idx, vi, vj, rel, step, trans_att) (not sorted)

        self.memorized_nodes = None  # (np.array) (eg_idx, v) sorted by ed_idx, v

        self.node_attention_li = None
        self.attend_nodes_li = None

    def _add_virtual_edges(self, n_clusters, n_clustering, connected_clustering, n_ents, n_rels):
        """ `n_ents`: all real nodes in `train` + `valid` + `test`
        """

        def add_edges(n_real, k, n_ents, has_center=False):
            k = min(n_real, k)
            in_clustering = np.array_split(np.random.permutation(n_real), k)
            out_clustering = np.array_split(np.random.permutation(n_real), k)
            virtual_nodes = [n_ents + i for i in range(k)]
            virtual_edges = []
            for virtual_v, in_clus, out_clus in zip(virtual_nodes, in_clustering, out_clustering):
                for real_v in in_clus:
                    virtual_edges.append((real_v, virtual_v, self.virtual_relations.real2virtual))
                for real_v in out_clus:
                    virtual_edges.append((virtual_v, real_v, self.virtual_relations.virtual2real))
                for other_virtual_v in virtual_nodes:
                    if virtual_v != other_virtual_v:
                        virtual_edges.append((virtual_v, other_virtual_v, self.virtual_relations.virtual2virtual))

            if has_center:
                center_v = n_ents + k
                for virtual_v in virtual_nodes:
                    virtual_edges.append((virtual_v, center_v, self.virtual_relations.virtual2vcenter))
                    virtual_edges.append((center_v, virtual_v, self.virtual_relations.vcenter2virtual))
                virtual_nodes.append(center_v)
                return virtual_nodes, virtual_edges, n_ents + k + 1, center_v
            else:
                return virtual_nodes, virtual_edges, n_ents + k, None

        virtual_nodes = []
        virtual_edges = []
        virtual_centers = []
        n_real = n_ents
        for t in range(n_clustering):
            vir_nodes, vir_edges, n_ents, center_v = add_edges(n_real, n_clusters, n_ents,
                                                               has_center=connected_clustering)
            virtual_nodes += vir_nodes
            virtual_edges += vir_edges
            virtual_centers.append(center_v)

        if connected_clustering:
            for c1 in virtual_centers:
                for c2 in virtual_centers:
                    if c1 != c2:
                        virtual_edges.append((c1, c2, self.virtual_relations.vcenter2vcenter))

        return virtual_nodes, virtual_edges

    def _count_over_full_edges(self, edges):
        dct = defaultdict(int)
        for i, h, t, r in edges:
            dct[h] += 1
            dct[(h, r)] += 1
            dct[(h, r, t)] += 1
        return dct

    def _make_edge2id(self, edges):
        dct = defaultdict(set)
        for i, h, t, r in edges:
            dct[(h, t, r)].add(i)
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
            for h, t, r in train:
                train_eids.update(self.edge2id[(h, t, r)])
            graph_eids = set()
            for id_set in self.edge2id.values():
                graph_eids.update(id_set)
            graph_eids = graph_eids - train_eids
            graph_eids = np.sort(np.array(list(graph_eids), dtype='int32'))
            self.edges = self.full_edges[graph_eids]
            self.n_edges = len(self.edges)
            yield train, self
            start = end

    def use_full_edges(self):
        self.edges = self.full_edges
        self.n_edges = len(self.edges)

    def get_candidate_edges(self, attended_nodes=None, tc=None):
        """ attended_nodes:
            (1) None: use all graph edges with batch_size=1
            (2) (np.array) n_attended_nodes x 2, (eg_idx, vi) sorted
        """
        if tc is not None:
            t0 = time.time()

        if attended_nodes is None:
            candidate_edges = np.concatenate([np.zeros((self.n_edges, 1), dtype='int32'),
                                              self.edges], axis=1)  # (0, edge_id, vi, vj, rel) sorted by (0, edge_id)

        else:
            candidate_idx, new_eg_idx = groupby_1cols_merge(attended_nodes[:, 0], attended_nodes[:, 1],
                                                            self.edges[:, 1], self.edges[:, 0])
            candidate_edges = np.concatenate([np.expand_dims(new_eg_idx, 1),
                                              self.full_edges[candidate_idx]], axis=1)  # (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)

        if tc is not None:
            tc['candi_e'] += time.time() - t0
        # candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel)
        #   sorted by (eg_idx, edge_id) or (eg_idx, vi, vj, rel)
        return candidate_edges

    def sample_edges(self, candidate_edges, edges_logits, mode=None,
                     max_edges_per_eg=None, max_edges_per_vi=None, tc=None):
        """ candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
            edges_logits: (tf.Variable) n_full_edges
        """
        assert mode is not None
        if tc is not None:
            t0 = time.time()

        edges_logits = edges_logits.numpy()  # n_full_edges
        edge_id = candidate_edges[:, 1]  # n_candidate_edges
        logits = edges_logits[edge_id]  # n_candidate_edges
        n_logits = len(logits)

        eps = 1e-20
        loglog_u = - tf.math.log(- tf.math.log(tf.random.uniform((n_logits,)) + eps) + eps)
        loglog_u = np.array(loglog_u, dtype='float32')
        logits = logits + loglog_u  # n_candidate_edges

        if mode == 'by_eg':
            assert max_edges_per_eg is not None
            sampled_edges = candidate_edges[:, 0]  # n_candidate_edges
            sampled_idx = groupby_1cols_nlargest(sampled_edges, logits, max_edges_per_eg)  # n_sampled_edges
            sampled_edges = np.concatenate([candidate_edges[sampled_idx],
                                            np.expand_dims(sampled_idx, 1)], axis=1)  # n_sampled_edges x 6

        elif mode == 'by_vi':
            assert max_edges_per_vi is not None
            sampled_edges = candidate_edges[:, [0, 2]]  # n_candidate_edges x 2
            sampled_idx = groupby_2cols_nlargest(sampled_edges, logits, max_edges_per_vi)  # n_sampled_edges
            sampled_edges = np.concatenate([candidate_edges[sampled_idx],
                                            np.expand_dims(sampled_idx, 1)], axis=1)  # n_sampled_edges x 6

        else:
            raise ValueError('Invalid `mode`')

        if tc is not None:
            tc['sampl_e'] += time.time() - t0
        # loglog_u: (np.array) n_candidate_edges
        # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
        return loglog_u, sampled_edges

    def get_selected_edges(self, sampled_edges, tc=None):
        """ sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
        """
        if tc is not None:
            t0 = time.time()

        idx_vi = get_segment_ids(sampled_edges[:, [0, 2]])
        _, idx_vj = np.unique(sampled_edges[:, [0, 3]], axis=0, return_inverse=True)
        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)

        # selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj] sorted by (eg_idx, vi, vj)
        selected_edges = np.concatenate([sampled_edges[:, [0, 2, 3, 4]], idx_vi, idx_vj], axis=1)

        if tc is not None:
            tc['sele_e'] += time.time() - t0
        return selected_edges

    def set_init_memorized_nodes(self, heads, tc=None):
        """ heads: batch_size
        """
        if tc is not None:
            t0 = time.time()

        batch_size = heads.shape[0]
        eg_idx = np.array(np.arange(batch_size), dtype='int32')
        self.memorized_nodes = np.stack([eg_idx, heads], axis=1)

        if tc is not None:
            tc['i_memo_v'] += time.time() - t0
        return self.memorized_nodes  # n_memorized_nodes (=batch_size) x 2, (eg_idx, v) sorted by (ed_idx, v)

    def get_topk_nodes(self, node_attention, max_nodes, tc=None):
        """ node_attention: (tf.Tensor) batch_size x n_nodes
        """
        if tc is not None:
            t0 = time.time()

        eps = 1e-20
        node_attention = node_attention.numpy()
        n_nodes = node_attention.shape[1]
        max_nodes = min(n_nodes, max_nodes)
        sorted_idx = np.argsort(-node_attention, axis=1)[:, :max_nodes]
        sorted_idx = np.sort(sorted_idx, axis=1)
        node_attention = np.take_along_axis(node_attention, sorted_idx, axis=1)  # sorted node attention
        mask = node_attention > eps
        eg_idx = np.repeat(np.expand_dims(np.arange(mask.shape[0]), 1), mask.shape[1], axis=1)[mask].astype('int32')
        vi = sorted_idx[mask].astype('int32')
        topk_nodes = np.stack([eg_idx, vi], axis=1)

        if tc is not None:
            tc['topk_v'] += time.time() - t0
        # topk_nodes: (np.array) n_topk_nodes x 2, (eg_idx, vi) sorted
        return topk_nodes

    def set_past_transitions(self):
        self.past_transitions = None  # (pd.DataFrame) (eg_idx, vi, vj, rel, step, trans_att) (not sorted)

    def set_node_attention_li(self, node_attention):
        self.node_attention_li = [node_attention.numpy()]

    def set_attended_nodes_li(self):
        self.attend_nodes_li = []

    def get_selfloop_and_backtrace(self, attended_nodes, attended_node_att, max_backtrace_edges, step,
                                   backtrace_decay=1., tc=None):
        """ attended_nodes: (np.array) n_attended_nodes x 2, (eg_idx, vi) not sorted
            attended_node_att: (tf.Tensor) n_attended_nodes
            step: the current step
        """
        if tc is not None:
            t0 = time.time()

        eg_idx, vi = attended_nodes[:, 0], attended_nodes[:, 1]
        selfloop_edges = np.stack([eg_idx, vi, vi, np.repeat(np.array(self.selfloop, dtype='int32'), eg_idx.shape[0])],
                                  axis=1)  # (eg_idx, vi, vi, selfloop)

        if self.past_transitions is None:
            return selfloop_edges, None

        attended_node_att = attended_node_att.numpy()  # n_attended_nodes
        attended_pd = pd.DataFrame({'eg_idx': attended_nodes[:, 0],
                                    'vj': attended_nodes[:, 1],
                                    'vj_att': attended_node_att})
        merged_pd = pd.merge(self.past_transitions, attended_pd, on=['eg_idx', 'vj'])
        # merged_pd: (eg_idx, vi, vj, rel, step, trans_att, vj_att, att)
        merged_pd['att'] = merged_pd['trans_att'] * merged_pd['vj_att'] * (backtrace_decay ** (step - merged_pd['step']))
        result_pd = merged_pd.sort_values(['att']).groupby(['eg_idx']).head(max_backtrace_edges)
        result = result_pd[['eg_idx', 'vj', 'vi']].sort_values(['eg_idx', 'vj', 'vi']).values
        # result: (eg_idx, vj, vi, backtrace)
        backtrace_edges = np.concatenate([result,
                                          np.expand_dims(np.repeat(np.array(self.backtrace, dtype='int32'),
                                                                   len(result)), 1)], axis=1)

        if tc is not None:
            tc['sl_bt'] += time.time() - t0
        return selfloop_edges, backtrace_edges  # (eg_idx, vi, vi, selfloop), (eg_idx, vj, vi, backtrace)

    def get_union_edges(self, scanned_edges, selfloope_edges, backtrace_edges, tc=None):
        """ scanned_edges: (np.array) n_scanned_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
            selfloop_edges: (np.array) n_selfloop_edges x 4 (eg_idx, vi, vi, selfloop)
            backtrace_edges: (np.array) n_backtrace_edges x 4 (eg_idx, vj, vi, backtrace)
        """
        if tc is not None:
            t0 = time.time()

        scanned_edges = scanned_edges[:, :4]  # (eg_idx, vi, vj, rel)
        n_scanned_edges = scanned_edges.shape[0]
        if backtrace_edges is None:
            all_edges = np.concatenate([scanned_edges, selfloope_edges], axis=0)
        else:
            all_edges = np.concatenate([scanned_edges, selfloope_edges, backtrace_edges], axis=0)
        sorted_idx = np.squeeze(np.argsort(all_edges.view('<i4,<i4,<i4,<i4'),
                                           order=['f0', 'f1', 'f2'], axis=0), 1).astype('int32')

        new_idx = np.argsort(sorted_idx).astype('int32')
        new_idx_for_edges_y = np.expand_dims(new_idx[:n_scanned_edges], 1)
        rest_idx = np.expand_dims(new_idx[n_scanned_edges:], 1)

        aug_scanned_edges = all_edges[sorted_idx]  # sorted by (eg_idx, vi, vj)
        idx_vi = get_segment_ids(aug_scanned_edges[:, [0, 1]])
        _, idx_vj = np.unique(aug_scanned_edges[:, [0, 2]], axis=0, return_inverse=True)
        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)
        aug_scanned_edges = np.concatenate([aug_scanned_edges, idx_vi, idx_vj], axis=1)

        if tc is not None:
            tc['union_e'] += time.time() - t0
        # aug_scanned_edges: n_aug_scanned_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        return aug_scanned_edges, new_idx_for_edges_y, rest_idx

    def add_transitions(self, seen_edges, trans_att, step, tc=None):
        """ seen_edges: n_seen_edges x 6 (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
            trans_att: n_seen_edges
        """
        if tc is not None:
            t0 = time.time()

        mask = (seen_edges[:, 3] != self.selfloop) * (seen_edges[:, 3] != self.backtrace)
        seen_edges = seen_edges[mask]
        trans_att = trans_att[mask]

        transitions = pd.DataFrame({'eg_idx': seen_edges[:, 0],
                                    'vi': seen_edges[:, 1],
                                    'vj': seen_edges[:, 2],
                                    'rel': seen_edges[:, 3],
                                    'step': np.repeat(np.array(step, dtype='int32'), len(seen_edges)),
                                    'trans_att': trans_att})

        if self.past_transitions is None:
            self.past_transitions = transitions
        else:
            self.past_transitions = pd.concat([self.past_transitions, transitions], ignore_index=True, sort=False)

        if tc is not None:
            tc['add_trans'] += time.time() - t0

    def add_node_attention(self, node_attention):
        self.node_attention_li.append(node_attention.numpy())

    def add_attended_nodes(self, attended_nodes):
        self.attend_nodes_li.append(attended_nodes)

    def add_nodes_to_memorized(self, selected_edges, inplace=False, tc=None):
        """ selected_edges: (np.array) n_selected_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        """
        if tc is not None:
            t0 = time.time()

        selected_vj = np.unique(selected_edges[:, [0, 2]], axis=0)
        mask = np.in1d(selected_vj.view('<i4,<i4'), self.memorized_nodes.view('<i4,<i4'), assume_unique=True)
        mask = np.logical_not(mask)
        new_nodes = selected_vj[mask]  # n_new_nodes x 2

        if len(new_nodes) > 0:
            memorized_and_new = np.concatenate([self.memorized_nodes, new_nodes], axis=0)  # n_memorized_and_new_nodes x 2
            sorted_idx = np.squeeze(np.argsort(memorized_and_new.view('<i4,<i4'),
                                               order=['f0', 'f1'], axis=0), 1).astype('int32')

            memorized_and_new = memorized_and_new[sorted_idx]
            n_memorized_and_new_nodes = len(memorized_and_new)

            new_idx = np.argsort(sorted_idx).astype('int32')
            n_memorized_nodes = self.memorized_nodes.shape[0]
            new_idx_for_memorized = np.expand_dims(new_idx[:n_memorized_nodes], 1)

            if inplace:
                self.memorized_nodes = memorized_and_new
        else:
            new_idx_for_memorized = None
            memorized_and_new = self.memorized_nodes
            n_memorized_and_new_nodes = len(memorized_and_new)

        if tc is not None:
            tc['add_scan'] += time.time() - t0
        # new_idx_for_memorized: n_memorized_nodes x 1
        # memorized_and_new: n_memorized_and_new_nodes x 2, (eg_idx, v) sorted by (eg_idx, v)
        return new_idx_for_memorized, n_memorized_and_new_nodes, memorized_and_new

    def set_index_over_nodes(self, selected_edges, nodes, tc=None):
        """ selected_edges (or aug_selected_edges): n_selected_edges (or n_aug_selected_edges) x 6, sorted
            nodes: (eg_idx, v) unique and sorted
        """
        if tc is not None:
            t0 = time.time()

        selected_vi = get_unique(selected_edges[:, [0, 1]])  # n_selected_edges x 2
        selected_vj = np.unique(selected_edges[:, [0, 2]], axis=0)  # n_selected_edges x 2
        mask_vi = np.in1d(nodes.view('<i4,<i4'), selected_vi.view('<i4,<i4'), assume_unique=True)
        mask_vj = np.in1d(nodes.view('<i4,<i4'), selected_vj.view('<i4,<i4'), assume_unique=True)
        new_idx_e2vi = np.expand_dims(np.arange(mask_vi.shape[0])[mask_vi], 1).astype('int32')  # n_matched_by_idx_and_vi x 1
        new_idx_e2vj = np.expand_dims(np.arange(mask_vj.shape[0])[mask_vj], 1).astype('int32')  # n_matched_by_idx_and_vj x 1

        idx_vi = selected_edges[:, 4]
        idx_vj = selected_edges[:, 5]
        new_idx_e2vi = new_idx_e2vi[idx_vi]  # n_selected_edges x 1
        new_idx_e2vj = new_idx_e2vj[idx_vj]  # n_selected_edges x 1

        # selected_edges: n_selected_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        selected_edges = np.concatenate([selected_edges, new_idx_e2vi, new_idx_e2vj], axis=1)

        if tc is not None:
            tc['idx_v'] += time.time() - t0
        return selected_edges

    def get_seen_edges(self, seen_nodes, aug_scanned_edges, tc=None):
        """ seen_nodes: (np.array) n_seen_nodes x 2, (eg_idx, vj) unique but not sorted
            aug_scanned_edges: (np.array) n_aug_scanned_edges x 8,
                (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        """
        if tc is not None:
            t0 = time.time()

        aug_scanned_vj = aug_scanned_edges[:, [0, 2]].copy()  # n_aug_scanned_edges x 2, (eg_idx, vj) not unique and not sorted
        mask_vj = np.in1d(aug_scanned_vj.view('<i4,<i4'), seen_nodes.view('<i4,<i4'))
        seen_edges = aug_scanned_edges[mask_vj][:, :4]  # n_seen_edges x 4, (eg_idx, vi, vj, rel) sorted by (eg_idx, vi, vj)

        seen_idx_for_edges_y = np.expand_dims(np.arange(mask_vj.shape[0])[mask_vj], 1).astype('int32')  # n_seen_edges x 1

        idx_vi = get_segment_ids(seen_edges[:, [0, 1]])
        _, idx_vj = np.unique(seen_edges[:, [0, 2]], axis=0, return_inverse=True)
        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)
        seen_edges = np.concatenate((seen_edges, idx_vi, idx_vj), axis=1)

        if tc is not None:
            tc['seen_e'] += time.time() - t0
        # seen_edges: n_seen_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        # seen_idx_for_edges_y: n_seen_edges x 1
        return seen_edges, seen_idx_for_edges_y


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
        self.graph = Graph(dataset.train, dataset.n_entities, dataset.n_relations, hparams)

        self.filter_pool = defaultdict(set)
        for head, tail, rel in np.concatenate([self.graph.full_train, self.valid_triples, self.test_triples], axis=0):
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
