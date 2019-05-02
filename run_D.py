import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from collections import defaultdict, namedtuple
from functools import partial
import time
import argparse
import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras

from utils import get, get_segment_ids, get_unique, groupby_2cols_nlargest, groupby_1cols_nlargest, \
    groupby_1cols_merge, groupby_1cols_cartesian
import datasets

tf.config.gpu.set_per_process_memory_growth(True)


""" dataenv.py """

VirtualRelations = namedtuple('VirtualRelations',
                              ['real2virtual', 'virtual2real', 'virtual2virtual',
                               'virtual2vcenter', 'vcenter2virtual', 'vcenter2vcenter'])

class Graph(object):
    def __init__(self, train_triples, n_ents, n_rels, reversed_rel_dct, hparams):
        """ `train_triples`: all (head, tail, rel) in `train`
            `n_ents`: all real nodes in `train` + `valid` + `test`
            `n_rels`: all real relations in `train` + `valid` + `test`

            Virtual nodes should connected to all real nodes in `train` + `valid` + `test`
            via two types of virtual relation: `into_virtual` and `outof_virtual`
        """
        self.train = train_triples
        self.n_train = len(self.train)
        self.reversed_rel_dct = reversed_rel_dct

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
        self.n_relations = self.n_relations + 2  # including 'selfloop' and 'backtrace'

        # `edges`: remove the current train triples
        # edges[i] = [id, head, tail, rel] sorted by head, tail, rel with ascending but not consecutive `id`s
        self.edges = None
        self.n_edges = 0

        self.memorized_nodes = None  # (np.array) (eg_idx, v) sorted by ed_idx, v
        self.memorized_node_atts = None  # (np.array) (v_att,) listed according to `memorized_nodes`

        self.inputs_history = []
        self.node_attention_history = []
        self.attend_nodes_history = []

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

    def make_temp_edges(self, batch):
        """ batch: (np.array) (head, tail, rel)
        """
        batch_set = set([(h, t, r) for h, t, r in batch])
        if self.reversed_rel_dct is None:
            self.edges = np.array([(eid, h, t, r)
                                   for eid, h, t, r in self.full_edges
                                   if (h, t, r) not in batch_set], dtype='int32')
            self.n_edges = len(self.edges)
        else:
            self.edges = np.array([(eid, h, t, r)
                                   for eid, h, t, r in self.full_edges
                                   if (h, t, r) not in batch_set and (t, h, self.reversed_rel_dct[r]) not in batch_set],
                                  dtype='int32')
            self.n_edges = len(self.edges)

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
                     max_edges_per_eg=None, max_edges_per_vi=None, use_gumbel_softmax=False, tc=None):
        """ candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
            edges_logits: (tf.Variable) n_full_edges
        """
        assert mode is not None
        if tc is not None:
            t0 = time.time()

        if use_gumbel_softmax:
            edges_logits = edges_logits.numpy()  # n_full_edges
            edge_id = candidate_edges[:, 1]  # n_candidate_edges
            logits = edges_logits[edge_id]  # n_candidate_edges
            n_logits = len(logits)

            eps = 1e-20
            loglog_u = - tf.math.log(- tf.math.log(tf.random.uniform((n_logits,)) + eps) + eps)
            loglog_u = np.array(loglog_u, dtype='float32')
            logits = logits + loglog_u  # n_candidate_edges
        else:
            logits = tf.random.uniform((len(candidate_edges),))
            loglog_u = None

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
        self.memorized_node_atts = np.ones((batch_size,))

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

    def add_inputs(self, inputs, epoch, batch_i):
        self.inputs_history.append((epoch, batch_i, inputs))

    def set_init(self, node_attention, epoch, batch_i):
        self.node_attention_history.append([epoch, batch_i, node_attention.numpy()])
        self.attend_nodes_history.append([epoch, batch_i])

    def get_selfloop_and_backtrace(self, attended_nodes, max_backtrace_nodes, tc=None):
        """ attended_nodes: (np.array) n_attended_nodes x 2, (eg_idx, vi) sorted
        """
        if tc is not None:
            t0 = time.time()

        eg_idx, vi = attended_nodes[:, 0], attended_nodes[:, 1]
        selfloop_edges = np.stack([eg_idx, vi, vi, np.repeat(np.array(self.selfloop, dtype='int32'), eg_idx.shape[0])],
                                  axis=1)  # (eg_idx, vi, vi, selfloop)

        if self.memorized_nodes is None or max_backtrace_nodes == 0:
            return selfloop_edges, None

        backtrace_idx = groupby_1cols_nlargest(self.memorized_nodes[:, 0], self.memorized_node_atts, max_backtrace_nodes)
        backtrace_nodes = self.memorized_nodes[backtrace_idx]  # (eg_idx, vj)
        backtrace_edges = groupby_1cols_cartesian(attended_nodes[:, 0], attended_nodes[:, 1],
                                                  backtrace_nodes[:, 0], backtrace_nodes[:, 1])
        # backtrace_edges: (eg_idx, vj, vi, backtrace)
        backtrace_edges = np.concatenate([backtrace_edges,
                                          np.expand_dims(np.repeat(np.array(self.backtrace, dtype='int32'),
                                                                   len(backtrace_edges)), 1)], axis=1)

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

    def add_flow(self, node_attention, attended_nodes):
        self.node_attention_history[-1].append(node_attention.numpy())
        self.attend_nodes_history[-1].append(attended_nodes)

    def add_nodes_to_memorized(self, selected_edges, node_attention=None, backtrace_decay=1., inplace=False, tc=None):
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
                assert node_attention is not None
                self.memorized_nodes = memorized_and_new
                cur_attts = tf.gather_nd(node_attention, memorized_and_new).numpy()
                prev_atts = tf.scatter_nd(new_idx_for_memorized, self.memorized_node_atts, (n_memorized_and_new_nodes,)).numpy()
                self.memorized_node_atts = cur_attts + prev_atts * backtrace_decay
        else:
            new_idx_for_memorized = None
            memorized_and_new = self.memorized_nodes
            n_memorized_and_new_nodes = len(memorized_and_new)

        if tc is not None:
            tc['add_scan'] += time.time() - t0
        # new_idx_for_memorized: n_memorized_nodes x 1
        # memorized_and_new: n_memorized_and_new_nodes x 2, (eg_idx, v) sorted by (eg_idx, v)
        return new_idx_for_memorized, n_memorized_and_new_nodes, memorized_and_new

    def get_index_from_nodes(self, query_nodes, nodes, tc=None):
        """ query_nodes: (eg_idx, v) unique and sorted by (eg_idx, v)
            nodes: (eg_idx, v) unique and sorted by (eg_idx, v)
        """
        if tc is not None:
            t0 = time.time()

        mask_v = np.in1d(nodes.view('<i4,<i4'), query_nodes.view('<i4,<i4'), assume_unique=True)
        idx = np.arange(mask_v.shape[0])[mask_v].astype('int32')

        if tc is not None:
            tc['get_idx'] += time.time() - t0
        return idx

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
    def get_train_batch(self, train_data, graph, batch_size, shuffle=True):
        n_train = len(train_data)
        rand_idx = np.random.permutation(n_train) if shuffle else np.arange(n_train)
        start = 0
        while start < n_train:
            end = min(start + batch_size, n_train)
            batch = np.array([train_data[i] for i in rand_idx[start:end]], dtype='int32')
            graph.make_temp_edges(batch)
            yield batch, end - start
            start = end

    def get_eval_batch(self, eval_data, graph, batch_size, shuffle=True):
        n_eval = len(eval_data)
        rand_idx = np.random.permutation(n_eval) if shuffle else np.arange(n_eval)
        start = 0
        while start < n_eval:
            end = min(start + batch_size, n_eval)
            batch = np.array([eval_data[i] for i in rand_idx[start:end]], dtype='int32')
            graph.use_full_edges()
            yield np.array(batch, dtype='int32'), end - start
            start = end


class DataEnv(object):
    def __init__(self, dataset, hparams):
        self.hparams = hparams
        self.data_feeder = DataFeeder()

        self.valid = dataset.valid
        self.test = dataset.test
        self.n_valid = len(self.valid)
        self.n_test = len(self.test)

        # `graph`:
        # (1) graph.entities: train + valid + test + virtual nodes
        # (2) graph.relations: train + valid + test + virtual relations
        # (3) graph.full_edges: train + virtual edges
        # (4) graph.edges: train - split_train + virtual edges
        self.graph = Graph(dataset.train, dataset.n_entities, dataset.n_relations, dataset.reversed_rel_dct, hparams)

        self.filter_pool = defaultdict(set)
        for head, tail, rel in np.concatenate([self.graph.train, self.valid, self.test], axis=0):
            self.filter_pool[(head, rel)].add(tail)

    def get_train_batcher(self):
        return partial(self.data_feeder.get_train_batch, self.graph.train, self.graph)

    def get_valid_batcher(self):
        return partial(self.data_feeder.get_eval_batch, self.valid, self.graph)

    def get_test_batcher(self):
        return partial(self.data_feeder.get_eval_batch, self.test, self.graph)


""" model.py """

class F(keras.layers.Layer):
    """ interact: e.g. [[0, 1], [0, 1, 2]]
    """
    def __init__(self, interact, n_dims, use_gate=False, n_conds=0, cond_mode=None, name=None):
        super(F, self).__init__(name=name)
        self.interact = interact
        self.n_dims = n_dims
        self.use_gate = use_gate
        self.n_conds = n_conds
        self.cond_mode = cond_mode

    def build(self, input_shape):
        n_interact = len(self.interact)
        self.proj_fns = [keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj_%d' % i)
                         for i in range(n_interact)]
        self.gate_fns = [keras.layers.Dense(self.n_dims, activation=tf.sigmoid, name='gate_%d' % i)
                         for i in range(n_interact)] if self.use_gate else None

        if self.n_conds > 0 and self.cond_mode is not None:
            if self.cond_mode == 'tanh':
                activation = tf.tanh
            elif self.cond_mode == 'square':
                activation = lambda x: tf.math.square(tf.tanh(x))
            else:
                raise ValueError('Invalid `cond_mode`')

            self.cond_fns = [keras.layers.Dense(self.n_dims, activation=activation, use_bias=False, name='cond_%d' % i)
                             for i in range(self.n_conds)]

        self.output_fn = keras.layers.Dense(self.n_dims, activation=tf.tanh, use_bias=False, name='output')

    def call(self, inputs, condition=None, training=None):
        """ inputs[i]: bs x ... x n_dims
            condition[i]: bs x ... x n_dims
        """
        conds = []
        if condition is not None and self.cond_mode is not None:
            assert self.n_conds == len(condition)
            for i, cond in enumerate(condition):
                conds.append(self.cond_fns[i](cond))

        outs = []
        for i, idxs in enumerate(self.interact):
            x = tf.concat([inputs[idx] for idx in idxs], axis=-1)
            out = self.proj_fns[i](x)
            if self.use_gate:
                gate = self.gate_fns[i](x)
                out = out * gate
            for cond in conds:
                out = out * cond

            outs.append(out)

        x = tf.concat(outs, axis=-1)
        out = self.output_fn(x)
        return out


class FF(keras.layers.Layer):
    def __init__(self, interact_1, interact_2, n_dims, use_gate=False, n_conds_1=0, n_conds_2=0, cond_mode=None,
                 name=None):
        super(FF, self).__init__(name=name)
        self.interact_1 = None if interact_1 is None or len(interact_1) == 0 else interact_1
        self.interact_2 = None if interact_2 is None or len(interact_2) == 0 else interact_2
        self.n_dims = n_dims
        self.use_gate = use_gate
        self.n_conds_1 = n_conds_1
        self.n_conds_2 = n_conds_2
        self.cond_mode = cond_mode

    def build(self, input_shape):
        self.f_1 = F(self.interact_1, self.n_dims, use_gate=self.use_gate,
                     n_conds=self.n_conds_1, cond_mode=self.cond_mode, name='f_1') \
            if self.interact_1 is not None else None

        self.f_2 = F(self.interact_2, self.n_dims, use_gate=self.use_gate,
                     n_conds=self.n_conds_2, cond_mode=self.cond_mode, name='f_2') \
            if self.interact_2 is not None else None

    def call(self, inputs, condition_1=None, condition_2=None, training=None):
        out = self.f_1(inputs, condition=condition_1) if self.f_1 is not None else inputs
        out = self.f_2(inputs + (out,), condition=condition_2) if self.f_2 is not None else out
        return out


class G(keras.layers.Layer):
    def __init__(self, interact, n_dims, name=None):
        """ interact: e.g. [[[0], [1]], [[0, 2], [1, 2]]]
        """
        super(G, self).__init__(name=name)
        self.interact = interact
        self.n_dims = n_dims

    def build(self, input_shape):
        n_interact = len(self.interact)
        self.left_proj_fns = [keras.layers.Dense(self.n_dims, activation=tf.tanh, name='left_proj_%d' % i)
                              for i in range(n_interact)]
        self.right_proj_fns = [keras.layers.Dense(self.n_dims, activation=tf.tanh, name='right_proj_%d' % i)
                               for i in range(n_interact)]

    def call(self, inputs, training=None):
        """ inputs[i]: bs x ... x n_dims
        """
        outs = []
        for i, (left_idxs, right_idxs) in enumerate(self.interact):
            left_x = tf.concat([inputs[idx] for idx in left_idxs], axis=-1)
            left_out = self.left_proj_fns[i](left_x)

            right_x = tf.concat([inputs[idx] for idx in right_idxs], axis=-1)
            right_out = self.right_proj_fns[i](right_x)

            out = tf.reduce_sum(left_out * right_out, axis=-1)
            outs.append(out)

        out = tf.add_n(outs)
        return out


def update_op(inputs, update):
    out = inputs + update
    return out


def node2edge_op(inputs, selected_edges, return_vi=True, return_vj=True):
    """ inputs (hidden): batch_size x n_nodes x n_dims
        selected_edges: n_selected_edges x 6 (or 8) ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
    """
    hidden = inputs
    batch_size = tf.shape(inputs)[0]
    n_selected_edges = len(selected_edges)
    idx = tf.cond(tf.equal(batch_size, 1), lambda: tf.zeros((n_selected_edges,), dtype='int32'), lambda: selected_edges[:, 0])
    result = []
    if return_vi:
        idx_and_vi = tf.stack([idx, selected_edges[:, 1]], axis=1)  # n_selected_edges x 2
        hidden_vi = tf.gather_nd(hidden, idx_and_vi)  # n_selected_edges x n_dims
        result.append(hidden_vi)
    if return_vj:
        idx_and_vj = tf.stack([idx, selected_edges[:, 2]], axis=1)  # n_selected_edges x 2
        hidden_vj = tf.gather_nd(hidden, idx_and_vj)  # n_selected_edges x n_dims
        result.append(hidden_vj)
    return result


def node2edge_v2_op(inputs, selected_edges, return_vi=True, return_vj=True):
    """ inputs (hidden): n_selected_nodes x n_dims
        selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
    """
    assert selected_edges is not None
    assert return_vi or return_vj
    hidden = inputs
    result = []
    if return_vi:
        new_idx_e2vi = selected_edges[:, 6]  # n_selected_edges
        hidden_vi = tf.gather(hidden, new_idx_e2vi)  # n_selected_edges x n_dims
        result.append(hidden_vi)
    if return_vj:
        new_idx_e2vj = selected_edges[:, 7]  # n_selected_edges
        hidden_vj = tf.gather(hidden, new_idx_e2vj)  # n_selected_edges x n_dims
        result.append(hidden_vj)
    return result


def aggregate_op(inputs, selected_edges, output_shape, at='vj', aggr_op='mean_v2'):
    """ inputs (edge_vec): n_seleted_edges x n_dims
        selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
        output_shape: (batch_size=1, n_nodes, n_dims)
    """
    assert selected_edges is not None
    assert output_shape is not None
    edge_vec = inputs
    if at == 'vi':
        idx_vi = selected_edges[:, 4]  # n_selected_edges
        aggr_op = tf.math.segment_mean if aggr_op == 'mean' or aggr_op == 'mean_v2' else \
            tf.math.segment_sum if aggr_op == 'sum' else \
            tf.math.segment_max if aggr_op == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vi)  # (max_idx_vi+1) x n_dims
        if aggr_op == 'mean_v2':
            edge_count = tf.math.segment_sum(tf.ones((len(idx_vi), 1)), idx_vi)  # (max_idx_vi+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1) - 1 + edge_count)  # (max_idx_vi+1) x n_dims
        idx_and_vi = tf.stack([selected_edges[:, 0], selected_edges[:, 1]], axis=1)  # n_selected_edges x 2
        idx_and_vi = tf.cast(tf.math.segment_max(idx_and_vi, idx_vi), tf.int32)  # (max_id_vi+1) x 2
        edge_vec_aggr = tf.scatter_nd(idx_and_vi, edge_vec_aggr, output_shape)  # batch_size x n_nodes x n_dims
    elif at == 'vj':
        idx_vj = selected_edges[:, 5]  # n_selected_edges
        max_idx_vj = tf.reduce_max(idx_vj)
        aggr_op = tf.math.unsorted_segment_mean if aggr_op == 'mean' or aggr_op == 'mean_v2' else \
            tf.math.unsorted_segment_sum if aggr_op == 'sum' else \
            tf.math.unsorted_segment_max if aggr_op == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x n_dims
        if aggr_op == 'mean_v2':
            edge_count = tf.math.unsorted_segment_sum(tf.ones((len(idx_vj), 1)), idx_vj)  # (max_idx_vj+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1) - 1 + edge_count)  # (max_idx_vj+1) x n_dims
        idx_and_vj = tf.stack([selected_edges[:, 0], selected_edges[:, 2]], axis=1)  # n_selected_edges x 2
        idx_and_vj = tf.cast(tf.math.unsorted_segment_max(idx_and_vj, idx_vj, max_idx_vj + 1), tf.int32)  # (max_idx_vj+1) x 2
        edge_vec_aggr = tf.scatter_nd(idx_and_vj, edge_vec_aggr, output_shape)  # batch_size x n_nodes x n_dims
    else:
        raise ValueError('Invalid `at`')
    return edge_vec_aggr


def aggregate_v2_op(inputs, selected_edges, output_shape, at='vj', aggr_op='mean_v2'):
    """ inputs (edge_vec): n_seleted_edges x n_dims
        selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
        output_shape: (n_visited_nodes, n_dims)
    """
    assert selected_edges is not None
    assert output_shape is not None
    edge_vec = inputs
    if at == 'vi':
        idx_vi = selected_edges[:, 4]  # n_selected_edges
        aggr_op = tf.math.segment_mean if aggr_op == 'mean' or aggr_op == 'mean_v2' else \
            tf.math.segment_sum if aggr_op == 'sum' else \
            tf.math.segment_max if aggr_op == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vi)  # (max_idx_vi+1) x n_dims
        if aggr_op == 'mean_v2':
            edge_count = tf.math.segment_sum(tf.ones((len(idx_vi), 1)), idx_vi)  # (max_idx_vi+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1) - 1 + edge_count)  # (max_idx_vi+1) x n_dims
        new_idx_e2vi = selected_edges[:, 6]  # n_selected_edges
        reduced_idx_e2vi = tf.cast(tf.math.segment_max(new_idx_e2vi, idx_vi), tf.int32)  # (max_id_vi+1)
        reduced_idx_e2vi = tf.expand_dims(reduced_idx_e2vi, 1)  # (max_id_vi+1) x 1
        edge_vec_aggr = tf.scatter_nd(reduced_idx_e2vi, edge_vec_aggr, output_shape)  # n_visited_nodes x n_dims
    elif at == 'vj':
        idx_vj = selected_edges[:, 5]  # n_selected_edges
        max_idx_vj = tf.reduce_max(idx_vj)
        aggr_op = tf.math.unsorted_segment_mean if aggr_op == 'mean' or aggr_op == 'mean_v2' else \
            tf.math.unsorted_segment_sum if aggr_op == 'sum' else \
            tf.math.unsorted_segment_max if aggr_op == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x n_dims
        if aggr_op == 'mean_v2':
            edge_count = tf.math.unsorted_segment_sum(tf.ones((len(idx_vj), 1)), idx_vj)  # (max_idx_vj+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1) - 1 + edge_count)  # (max_idx_vj+1) x n_dims
        new_idx_e2vj = selected_edges[:, 7]  # n_selected_edges
        reduced_idx_e2vj = tf.cast(tf.math.unsorted_segment_max(new_idx_e2vj, idx_vj, max_idx_vj + 1), tf.int32)  # (max_idx_vj+1)
        reduced_idx_e2vj = tf.expand_dims(reduced_idx_e2vj, 1)  # (max_idx_vj+1) x 1
        edge_vec_aggr = tf.scatter_nd(reduced_idx_e2vj, edge_vec_aggr, output_shape)  # n_visited_nodes x n_dims
    else:
        raise ValueError('Invalid `at`')
    return edge_vec_aggr


def sparse_softmax_op(logits, segment_ids, diff_control=None, sort=True):
    if diff_control == 'tanh':
        logits = tf.tanh(logits)
    if sort:
        logits_max = tf.math.segment_max(logits, segment_ids)
        logits_max = tf.gather(logits_max, segment_ids)
        logits_diff = logits - logits_max
        if isinstance(diff_control, float):
            logits_diff_max = tf.stop_gradient(tf.math.segment_max(tf.abs(logits_diff), segment_ids))
            logits_diff_max = tf.gather(tf.maximum(logits_diff_max / diff_control, 1), segment_ids)
            logits_exp = tf.math.exp(logits_diff / logits_diff_max)
        else:
            logits_exp = tf.math.exp(logits_diff)
        logits_expsum = tf.math.segment_sum(logits_exp, segment_ids)
        logits_expsum = tf.gather(logits_expsum, segment_ids)
        logits_norm = logits_exp / logits_expsum
    else:
        num_segments = tf.reduce_max(segment_ids) + 1
        logits_max = tf.math.unsorted_segment_max(logits, segment_ids, num_segments)
        logits_max = tf.gather(logits_max, segment_ids)
        logits_diff = logits - logits_max
        if isinstance(diff_control, float):
            logits_diff_max = tf.stop_gradient(tf.math.unsorted_segment_max(tf.abs(logits_diff), segment_ids, num_segments))
            logits_diff_max = tf.gather(tf.maximum(logits_diff_max / diff_control, 1), segment_ids)
            logits_exp = tf.math.exp(logits_diff / logits_diff_max)
        else:
            logits_exp = tf.math.exp(logits_diff)
        logits_expsum = tf.math.unsorted_segment_sum(logits_exp, segment_ids, num_segments)
        logits_expsum = tf.gather(logits_expsum, segment_ids)
        logits_norm = logits_exp / logits_expsum
    return logits_norm


def neighbor_softmax_op(inputs, selected_edges, at='vi', diff_control=None):
    """ inputs (edge_vec): n_seleted_edges x ...
        selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
    """
    assert selected_edges is not None
    edge_vec = inputs
    if at == 'vi':
        idx_vi = selected_edges[:, 4]  # n_selected_edges
        edge_vec_norm = sparse_softmax_op(edge_vec, idx_vi, diff_control=diff_control)  # n_selected_edges x ...
    elif at == 'vj':
        idx_vj = selected_edges[:, 5]  # n_selected_edges
        edge_vec_norm = sparse_softmax_op(edge_vec, idx_vj, diff_control=diff_control, sort=False)  # n_selected_edges x ...
    else:
        raise ValueError('Invalid `at`')
    return edge_vec_norm


class Sampler(keras.Model):
    def __init__(self, graph, use_gumbel_softmax, probability_mode, temperature):
        super(Sampler, self).__init__(name='sampler')
        self.graph = graph
        self.use_gumbel_softmax = use_gumbel_softmax
        self.probability_mode = probability_mode
        self.temperature = temperature

        if self.use_gumbel_softmax:
            with tf.name_scope(self.name):
                with tf.device('/cpu:0'):
                    if probability_mode == 'by_edge':
                        self.prob_params = self.add_weight(shape=(graph.n_full_edges,),
                                                           initializer=keras.initializers.zeros(),
                                                           name='prob_params')  # n_full_edges
                    elif probability_mode == 'by_rel':
                        self.prob_params = self.add_weight(shape=(graph.n_relations,),
                                                           initializer=keras.initializers.zeros(),
                                                           name='prob_params')  # n_relations
                    else:
                        raise ValueError('Invalid `probability_mode`')

    @property
    def edges_logits(self):
        self._edges_logits = None
        if self.use_gumbel_softmax:
            if self.probability_mode == 'by_edge':
                self._edges_logits = self.prob_params
            elif self.probability_mode == 'by_rel':
                rel_idx = self.graph.full_edges[:, 3]
                self._edges_logits = tf.gather(self.prob_params, rel_idx)
        return self._edges_logits

    def call(self, _, candidate_edges=None, loglog_u=None, sampled_edges=None, mode=None, diff_control=None,
             training=None, tc=None):
        """ inputs: None
            candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
            loglog_u: (np.array) n_candidate_edges
            sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
        """
        assert candidate_edges is not None
        assert sampled_edges is not None
        assert mode is not None
        if tc is not None:
            t0 = time.time()

        if self.use_gumbel_softmax:
            if mode == 'by_eg':
                segment_ids = candidate_edges[:, 0]  # n_candidate_edges

            elif mode == 'by_vi':
                segment_ids = get_segment_ids(candidate_edges[:, [0, 2]])

            else:
                raise ValueError('Invalid `mode`')

            edge_id = candidate_edges[:, 1]
            logits = tf.gather(self.edges_logits, edge_id)  # n_candidate_edges
            ca_idx = sampled_edges[:, 5]  # n_sampled_edges

            edges_y = self._gumbel_softmax(logits, loglog_u, segment_ids, ca_idx,
                                           temperature=self.temperature, diff_control=diff_control)

        else:
            edges_y = tf.ones((len(sampled_edges),))

        if tc is not None:
            tc['s.call'] += time.time() - t0
        return edges_y  # n_sampled_edges

    def _gumbel_softmax(self, logits, loglog_u, segment_ids, ca_idx, temperature=1., diff_control=None, hard=True):
        y = logits + loglog_u  # n_candidate_edges
        y = sparse_softmax_op(y, segment_ids, diff_control=diff_control)  # n_candidate_edges
        y = tf.gather(y / temperature, ca_idx)  # n_sampled_edges
        if hard:
            y_hard = tf.ones_like(y)
            y = tf.stop_gradient(y_hard - y) + y
        return y  # n_sampled_edges


class SharedEmbedding(keras.Model):
    def __init__(self, n_entities, n_relations, n_dims_lg, ent_emb_l2, rel_emb_l2, use_ent_emb):
        """ n_entities: including virtual nodes
            n_relations: including 'virtual', 'selfloop' and 'backtrace'
        """
        super(SharedEmbedding, self).__init__(name='shared_emb')
        self.n_dims = n_dims_lg
        self.use_ent_emb = use_ent_emb

        if self.use_ent_emb:
            self.entity_embedding = keras.layers.Embedding(n_entities, self.n_dims,
                                                           embeddings_regularizer=keras.regularizers.l2(ent_emb_l2),
                                                           name='entities')  # n_nodes x n_dims_lg
        self.relation_embedding = keras.layers.Embedding(n_relations, self.n_dims,
                                                         embeddings_regularizer=keras.regularizers.l2(rel_emb_l2),
                                                         name='relations')  # n_rels x n_dims_lg

    def call(self, inputs, target=None, training=None):
        assert target is not None
        if target == 'entity':
            return self.entity_embedding(inputs) if self.use_ent_emb \
                else tf.zeros(tf.concat((tf.shape(inputs), (self.n_dims,)), 0))
        elif target == 'relation':
            return self.relation_embedding(inputs)
        else:
            raise ValueError('Invalid `target`')


class QueryContext(keras.Model):
    def __init__(self, n_dims_lg):
        super(QueryContext, self).__init__(name='query_context')
        self.n_dims = n_dims_lg

    def call(self, inputs, shared_embedding=None, training=None, **kwargs):
        """ inputs: (heads, rels)
                heads: batch_size
                rels: batch_size
        """
        assert shared_embedding is not None
        with tf.name_scope(self.name):
            heads, rels = inputs
            head_emb = shared_embedding(heads, target='entity')  # batch_size x n_dims_lg
            rel_emb = shared_embedding(rels, target='relation')  # batch_size x n_dims_lg
        return head_emb, rel_emb


class UnconsciousnessFlow(keras.Model):
    def __init__(self, n_entities, n_dims_lg,
                 use_gate,
                 message_cond_mode,
                 hidden_cond_mode,
                 message_interact,
                 hidden_interact,
                 use_ent_emb):

        super(UnconsciousnessFlow, self).__init__(name='uncon_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims_lg
        self.use_ent_emb = use_ent_emb

        # fn(hidden_vi, rel_emb, hidden_vj) (vi -> vj: non-symmetric)
        self.message_fn = F(message_interact, self.n_dims, use_gate=use_gate, n_conds=1, cond_mode=message_cond_mode,
                            name='message_fn')

        # fn(message_aggr, hidden, ent_emb)
        self.hidden_fn = F(hidden_interact, self.n_dims, use_gate=use_gate, n_conds=1, cond_mode=hidden_cond_mode,
                           name='hidden_fn')

    def call(self, inputs, selected_edges=None, edges_y=None, shared_embedding=None, training=None, tc=None):
        """ inputs (hidden): 1 x n_nodes x n_dims_lg
            selected_edges: n_selected_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
            edges_y: n_selected_edges

            Here: batch_size = 1
        """
        assert selected_edges is not None
        assert edges_y is not None
        assert shared_embedding is not None
        if tc is not None:
            t0 = time.time()

        # compute unconscious messages
        hidden = inputs
        hidden_vi, hidden_vj = node2edge_op(hidden, selected_edges)  # n_selected_edges x n_dims_lg
        rel_idx = selected_edges[:, 3]  # n_selected_edges
        rel_emb = shared_embedding(rel_idx, target='relation')  # n_selected_edges x n_dims_lg
        message = self.message_fn((hidden_vi, rel_emb, hidden_vj), condition=[hidden_vi])  # n_selected_edges x n_dims_lg
        message = tf.expand_dims(edges_y, 1) * message  # n_selected_edges x n_dims_lg

        # aggregate unconscious messages
        message_aggr = aggregate_op(message, selected_edges, (1, self.n_nodes, self.n_dims))  # 1 x n_nodes x n_dims_lg

        # update unconscious states
        if self.use_ent_emb:
            ent_idx = tf.expand_dims(tf.range(0, self.n_nodes), axis=0)  # 1 x n_nodes
            ent_emb = shared_embedding(ent_idx, target='entity')  # 1 x n_nodes x n_dims_lg
            update = self.hidden_fn((message_aggr, hidden, ent_emb), condition=[message_aggr])  # 1 x n_nodes x n_dims_lg
        else:
            update = self.hidden_fn((message_aggr, hidden), condition=[message_aggr])  # 1 x n_nodes x n_dims_lg
        hidden = update_op(hidden, update)  # 1 x n_nodes x n_dims_lg

        if tc is not None:
            tc['u.call'] += time.time() - t0
        return hidden  # 1 x n_nodes x n_dims_lg

    def get_init_hidden(self, shared_embedding):
        with tf.name_scope(self.name):
            if self.use_ent_emb:
                ent_idx = tf.expand_dims(tf.range(0, self.n_nodes), axis=0)  # 1 x n_nodes
                ent_emb = shared_embedding(ent_idx, target='entity')  # 1 x n_nodes x n_dims_lg
                hidden = ent_emb
            else:
                hidden = tf.zeros((1, self.n_nodes, self.n_dims))
        return hidden


class ConsciousnessFlow(keras.Model):
    def __init__(self, n_entities, n_dims,
                 use_gate,
                 message_cond_mode,
                 hidden_cond_mode,
                 message_interacts,
                 hidden_interacts,
                 straight_through_trans,
                 straight_through_node,
                 direct_intervention_trans,
                 direct_intervention_trans_mode,
                 direct_intervention_node,
                 direct_intervention_node_mode,
                 use_ent_emb):

        super(ConsciousnessFlow, self).__init__(name='con_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims
        self.straight_through_trans = straight_through_trans
        self.straight_through_node = straight_through_node
        self.direct_intervention_trans = direct_intervention_trans
        self.direct_intervention_trans_mode = direct_intervention_trans_mode
        self.direct_intervention_node = direct_intervention_node
        self.direct_intervention_node_mode = direct_intervention_node_mode
        self.use_ent_emb = use_ent_emb

        # fn(hidden_vi, rel_emb, hidden_vj, query_head_emb or query_head_hidden, query_rel_emb)
        self.message_fn = FF(message_interacts[0], message_interacts[1], self.n_dims, use_gate=use_gate,
                             n_conds_1=1, n_conds_2=0, cond_mode=message_cond_mode, name='message_fn')

        # # fn(message_aggr, hidden, hidden_uncon, query_head_emb or query_head_hidden, query_rel_emb)
        self.hidden_fn = FF(hidden_interacts[0], hidden_interacts[1], self.n_dims, use_gate=use_gate,
                            n_conds_1=1, n_conds_2=0, cond_mode=hidden_cond_mode, name='hidden_fn')

        self.proj_for_rel_lg = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj_for_rel_lg')
        self.proj_for_hidden_lg = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj_for_hidden_lg')

        if self.direct_intervention_trans:
            if self.direct_intervention_trans_mode == 'scaled':
                self.interven_trans = keras.layers.Dense(self.n_dims, use_bias=False, name='interven_trans')
            elif self.direct_intervention_trans_mode == 'tanh':
                self.interven_trans = keras.layers.Dense(self.n_dims, activation=tf.tanh, use_bias=False, name='interven_trans')

        if self.direct_intervention_node:
            if self.direct_intervention_node_mode == 'scaled':
                self.interven_node = keras.layers.Dense(self.n_dims, use_bias=False, name='interven_node')
            elif self.direct_intervention_node_mode == 'tanh':
                self.interven_node = keras.layers.Dense(self.n_dims, activation=tf.tanh, use_bias=False, name='interven_node')

    def call(self, inputs, seen_edges=None, edges_y=None, trans_attention=None, node_attention=None,
             hidden_uncon=None, memorized_nodes=None, query_head_emb=None, query_rel_emb=None, query_head_idx=None,
             shared_embedding=None, training=None, tc=None):
        """ inputs (hidden): n_memorized_nodes x n_dims
            seen_edges: n_seen_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by (idx, vi, vj)
                (1) including selfloop edges and backtrace edges
                (2) batch_size >= 1
            edges_y: n_seen_edges ( sorted according to seen_edges )
            trans_attention: n_seen_edges ( sorted according to seen_edges )
            node_attention: batch_size x n_nodes
            hidden_uncon: 1 x n_nodes x n_dims_lg
            memorized_nodes: n_memorized_nodes x 2, (eg_idx, v)
            query_head_emb: batch_size x n_dims_lg
            query_rel_emb: batch_size x n_dims_lg
            query_heads: batch_size
        """
        assert seen_edges is not None
        assert edges_y is not None
        assert trans_attention is not None
        assert node_attention is not None
        assert hidden_uncon is not None
        assert query_head_emb is not None or query_head_idx is not None
        assert query_rel_emb is not None
        assert shared_embedding is not None
        if tc is not None:
            t0 = time.time()

        hidden = inputs

        if self.use_ent_emb:
            query_head_vec = self.proj_for_hidden_lg(query_head_emb)  # batch_size x n_dims
        else:
            query_head_vec = tf.gather(hidden, query_head_idx)  # batch_size x n_dims

        query_rel_vec = self.proj_for_rel_lg(query_rel_emb)  # batch_size x n_dims

        # compute conscious messages
        hidden_vi, hidden_vj = node2edge_v2_op(hidden, seen_edges)  # n_seen_edges x n_dims
        rel_idx = seen_edges[:, 3]  # n_seen_edges
        rel_emb = shared_embedding(rel_idx, target='relation')  # n_seen_edges x n_dims_lg
        rel_emb = self.proj_for_rel_lg(rel_emb)  # n_seen_edges x n_dims
        eg_idx = seen_edges[:, 0]  # n_seen_edges
        q_head_vec = tf.gather(query_head_vec, eg_idx)  # n_seen_edges x n_dims
        q_rel_vec = tf.gather(query_rel_vec, eg_idx)  # n_seen_edges x n_dims

        message = self.message_fn((hidden_vi, rel_emb, hidden_vj, q_head_vec, q_rel_vec), condition_1=[hidden_vi])  # n_seen_edges x n_dims
        message = tf.expand_dims(edges_y, 1) * message  # n_seen_edges x n_dims

        # attend conscious messages (hard, straight throught)
        if self.direct_intervention_trans:
            if self.direct_intervention_trans_mode == 'multiply':
                message = tf.expand_dims(trans_attention, 1) * message  # n_seen_edges x n_dims
            elif self.direct_intervention_trans_mode == 'scaled' or self.direct_intervention_trans_mode == 'tanh':
                message = self.interven_trans(tf.expand_dims(trans_attention, 1)) * message  # n_seen_edges x n_dims
        elif self.straight_through_trans:
            trans_attention = tf.stop_gradient(tf.ones_like(trans_attention) - trans_attention) + trans_attention  # n_seen_edges
            message = tf.expand_dims(trans_attention, 1) * message  # n_seen_edges x n_dims

        # aggregate conscious messages
        n_memorized_nodes = tf.shape(hidden)[0]
        message_aggr = aggregate_v2_op(message, seen_edges, (n_memorized_nodes, self.n_dims))  # n_memorized_nodes x n_dims

        # get unconscious states
        eg_idx, v = memorized_nodes[:, 0], memorized_nodes[:, 1]  # n_memorized_nodes, n_memorized_nodes
        hidden_uncon = tf.squeeze(hidden_uncon, axis=0)  # n_nodes x n_dims_lg
        hidden_uncon = tf.gather(hidden_uncon, v)  # n_memorized_nodes x n_dims_lg
        hidden_uncon = self.proj_for_hidden_lg(hidden_uncon)  # n_memorized_nodes x n_dims
        q_head_vec = tf.gather(query_head_vec, eg_idx)  # n_memorized_nodes x n_dims
        q_rel_vec = tf.gather(query_rel_vec, eg_idx)  # n_memorized_nodes x n_dims

        # attend unconscious states (hard, straight throught)
        if self.direct_intervention_node:
            idx_and_v = tf.stack([eg_idx, v], axis=1)  # n_memorized_nodes x 2
            node_attention = tf.gather_nd(node_attention, idx_and_v)  # n_memorized_nodes
            if self.direct_intervention_node_mode == 'multiply':
                hidden_uncon = tf.expand_dims(node_attention, 1) * hidden_uncon  # n_memorized_nodes x n_dims
            elif self.direct_intervention_node_mode == 'scaled' or self.direct_intervention_node_mode == 'tanh':
                hidden_uncon = self.interven_node(tf.expand_dims(node_attention, 1)) * hidden_uncon  # n_memorized_nodes x n_dims
        elif self.straight_through_node:
            idx_and_v = tf.stack([eg_idx, v], axis=1)  # n_memorized_nodes x 2
            node_attention = tf.gather_nd(node_attention, idx_and_v)  # n_memorized_nodes
            node_attention = tf.stop_gradient(tf.ones_like(node_attention) - node_attention) + node_attention  # n_memorized_nodes
            hidden_uncon = tf.expand_dims(node_attention, 1) * hidden_uncon  # n_memorized_nodes x n_dims

        # update conscious state
        update = self.hidden_fn((message_aggr, hidden, hidden_uncon, q_head_vec, q_rel_vec), condition_1=[message_aggr])  # n_memorized_nodes x n_dims
        hidden = update_op(hidden, update)  # n_memorized_nodes x n_dims

        if tc is not None:
            tc['c.call'] += time.time() - t0
        return hidden  # n_memorized_nodes x n_dims

    def get_init_hidden(self, hidden_uncon, memorized_nodes):
        """ hidden_uncon: 1 x n_nodes x n_dims_lg
            memorized_nodes: n_memorized_nodes (=batch_size) x 2, (eg_idx, v)
        """
        with tf.name_scope(self.name):
            idx, v = memorized_nodes[:, 0], memorized_nodes[:, 1]  # n_memorized_nodes, n_memorized_nodes
            hidden_uncon = tf.squeeze(hidden_uncon, axis=0)  # n_nodes x n_dims_lg
            hidden_uncon = tf.gather(hidden_uncon, v)  # n_memorized_nodes x n_dims_lg
            hidden_uncon = self.proj_for_hidden_lg(hidden_uncon)  # n_memorized_nodes x n_dims
            hidden_init = hidden_uncon
        return hidden_init  # n_memorized_nodes x n_dims


class AttentionFlow(keras.Model):
    def __init__(self, n_entities, n_dims_sm, transition_interact, use_ent_emb):
        super(AttentionFlow, self).__init__(name='att_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims_sm
        self.use_ent_emb = use_ent_emb

        # fn(hidden_con_vi, rel_emb, hidden_con_vj, hidden_uncon_vj, query_head_emb, query_rel_emb)
        self.transition_fn = G(transition_interact, self.n_dims, name='transition_fn')

        self.proj_for_hidden = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj_for_hidden')
        self.proj_for_hidden_lg = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj_for_hidden_lg')
        self.proj_for_rel_lg = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj_for_rel_lg')

    def call(self, inputs, scanned_edges=None, edges_y=None, hidden_uncon=None, hidden_con=None,
             new_idx_for_memorized=None, n_memorized_and_scanned_nodes=None,
             query_head_emb=None, query_rel_emb=None, query_head_idx=None, shared_embedding=None,
             diff_control=None, training=None, tc=None):
        """ inputs (node_attention): batch_size x n_nodes
            scanned_edges (aug_scanned_edges): n_aug_scanned_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
              (1) including selfloop edges and backtrace edges
              (2) batch_size >= 1
            edges_y: n_aug_scanned_edges ( sorted according to scanned_edges )
            hidden_uncon: 1 x n_nodes x n_dims_lg
            hidden_con: n_memorized_nodes x n_dims
            query_head_emb: batch_size x n_dims_lg
            query_rel_emb: batch_size x n_dims_lg
        """
        assert scanned_edges is not None
        assert edges_y is not None
        assert hidden_con is not None
        assert hidden_uncon is not None
        assert query_head_emb is not None
        assert query_rel_emb is not None or query_head_idx is not None
        assert n_memorized_and_scanned_nodes is not None
        assert shared_embedding is not None
        if tc is not None:
            t0 = time.time()

        hidden_con = self.proj_for_hidden(hidden_con)  # n_memorized_nodes x n_dims_sm
        if new_idx_for_memorized is not None:
            hidden_con = tf.scatter_nd(new_idx_for_memorized, hidden_con,
                                       tf.TensorShape((n_memorized_and_scanned_nodes, self.n_dims)))  # n_memorized_and_scanned_nodes x n_dims_sm
        hidden_uncon = self.proj_for_hidden_lg(hidden_uncon)  # 1 x n_nodes x n_dims_sm

        if self.use_ent_emb:
            query_head_vec = self.proj_for_hidden_lg(query_head_emb)  # batch_size x n_dims_sm
        else:
            query_head_vec = tf.gather(hidden_con, query_head_idx)  # batch_size x n_dims_sm

        query_rel_vec = self.proj_for_rel_lg(query_rel_emb)  # batch_size x n_dims_sm

        # compute transition
        hidden_con_vi, hidden_con_vj = node2edge_v2_op(hidden_con, scanned_edges)  # n_aug_scanned_edges x n_dims_sm
        hidden_uncon_vj, = node2edge_op(hidden_uncon, scanned_edges, return_vi=False)  # n_aug_scanned_edges x n_dims_sm
        rel_idx = scanned_edges[:, 3]  # n_aug_scanned_edges
        rel_emb = shared_embedding(rel_idx, target='relation')  # n_aug_scanned_edges x n_dims_lg
        rel_emb = self.proj_for_rel_lg(rel_emb)  # n_aug_scanned_edges x n_dims_sm

        eg_idx = scanned_edges[:, 0]  # n_aug_scanned_edges
        q_head_vec = tf.gather(query_head_vec, eg_idx)  # n_seen_edges x n_dims
        q_rel_vec = tf.gather(query_rel_vec, eg_idx)  # n_seen_edges x n_dims

        transition_logits = self.transition_fn((hidden_con_vi, rel_emb, hidden_con_vj, hidden_uncon_vj,
                                                q_head_vec, q_rel_vec))  # n_aug_scanned_edges
        transition = neighbor_softmax_op(transition_logits, scanned_edges, diff_control=diff_control)  # n_aug_scanned_edges

        # compute transition attention
        node_attention = inputs  # batch_size x n_nodes
        idx_and_vi = tf.stack([scanned_edges[:, 0], scanned_edges[:, 1]], axis=1)  # n_aug_scanned_edges x 2
        gathered_node_attention = tf.gather_nd(node_attention, idx_and_vi)  # n_aug_scanned_edges
        trans_attention = gathered_node_attention * transition * edges_y  # n_aug_scanned_edges

        # compute new node attention
        batch_size = tf.shape(inputs)[0]
        new_node_attention = aggregate_op(trans_attention, scanned_edges, (batch_size, self.n_nodes), aggr_op='sum')  # batch_size x n_nodes

        new_node_attention_sum = tf.reduce_sum(new_node_attention, axis=1, keepdims=True)  # batch_size x 1
        new_node_attention = new_node_attention / new_node_attention_sum  # batch_size x n_nodes

        if tc is not None:
            tc['a.call'] += time.time() - t0
        # trans_attention: n_aug_scanned_edges
        # new_node_attention: batch_size x n_nodes
        return trans_attention, new_node_attention

    def get_init_node_attention(self, heads, tc=None):
        with tf.name_scope(self.name):
            node_attention = tf.one_hot(heads, self.n_nodes)  # batch_size x n_nodes
        return node_attention


class Model(object):
    def __init__(self, graph, hparams):
        self.graph = graph
        self.hparams = hparams

        self.sampler = Sampler(graph,
                               self.hparams.use_gumbel_softmax,
                               self.hparams.probability_mode,
                               self.hparams.temperature)

        self.shared_embedding = SharedEmbedding(graph.n_entities, graph.n_relations, hparams.n_dims_lg,
                                                hparams.ent_emb_l2,
                                                hparams.rel_emb_l2,
                                                hparams.use_ent_emb)

        self.query_context = QueryContext(hparams.n_dims_lg)

        self.uncon_flow = UnconsciousnessFlow(graph.n_entities, hparams.n_dims_lg,
                                              hparams.use_gate,
                                              hparams.uncon_message_cond_mode,
                                              hparams.uncon_hidden_cond_mode,
                                              hparams.uncon_message_interact,
                                              hparams.uncon_hidden_interact,
                                              hparams.use_ent_emb)

        self.con_flow = ConsciousnessFlow(graph.n_entities, hparams.n_dims,
                                          hparams.use_gate,
                                          hparams.con_message_cond_mode,
                                          hparams.con_hidden_cond_mode,
                                          hparams.con_message_interacts,
                                          hparams.con_hidden_interacts,
                                          hparams.straight_through_trans,
                                          hparams.straight_through_node,
                                          hparams.direct_intervention_trans,
                                          hparams.direct_intervention_trans_mode,
                                          hparams.direct_intervention_node,
                                          hparams.direct_intervention_node_mode,
                                          hparams.use_ent_emb)

        self.att_flow = AttentionFlow(graph.n_entities, hparams.n_dims_sm,
                                      hparams.att_transition_interact,
                                      hparams.use_ent_emb)

        self.heads, self.rels = None, None

    def get_hparam_by_epoch(self, hparam, epoch):
        if not isinstance(hparam, (list, tuple)):
            return hparam
        else:
            return hparam[int(len(hparam) * (epoch - 1) / self.hparams.max_epochs)]

    def initialize(self, heads, rels, epoch, batch_i, tc=None):
        """ heads: batch_size
            rels: batch_size
        """
        self.heads = heads
        self.rels = rels
        query_head_emb, query_rel_emb = self.query_context((heads, rels), shared_embedding=self.shared_embedding)  # batch_size x n_dims_lg

        ''' initialize unconsciousness flow'''
        hidden_uncon = self.uncon_flow.get_init_hidden(self.shared_embedding)  # 1 x n_nodes x n_dims_lg

        ''' run unconsciousness flow before running consciousness flow '''
        if self.hparams.init_uncon_steps is not None:
            for _ in range(self.hparams.init_uncon_steps):
                # candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
                candidate_edges = self.graph.get_candidate_edges(tc=get(tc, 'graph'))

                # loglog_u: (np.array) n_candidate_edges
                # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
                loglog_u, sampled_edges = self.graph.sample_edges(candidate_edges, self.sampler.edges_logits,
                                                                  mode='by_eg',
                                                                  max_edges_per_eg=self.hparams.max_edges_per_example,
                                                                  use_gumbel_softmax=self.hparams.use_gumbel_softmax,
                                                                  tc=get(tc, 'graph'))

                # edges_y: (tf.Tensor) n_sample_edges
                diff_control = self.get_hparam_by_epoch(self.hparams.diff_control_for_sampling, epoch)
                edges_y = self.sampler(None, candidate_edges=candidate_edges, loglog_u=loglog_u,
                                       sampled_edges=sampled_edges, mode='by_eg', diff_control=diff_control,
                                       tc=get(tc, 'model'))

                # selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj] sorted by (eg_idx, vi, vj)
                selected_edges = self.graph.get_selected_edges(sampled_edges, tc=get(tc, 'graph'))

                hidden_uncon = self.uncon_flow(hidden_uncon, selected_edges=selected_edges, edges_y=edges_y,
                                               shared_embedding=self.shared_embedding,
                                               tc=get(tc, 'model'))  # 1 x n_nodes x n_dims_lg

        ''' initialize attention flow '''
        node_attention = self.att_flow.get_init_node_attention(heads)  # batch_size x n_nodes

        ''' initialize consciousness flow '''
        memorized_v = self.graph.set_init_memorized_nodes(heads)  # n_memorized_nodes (=batch_size) x 2, (eg_idx, v)
        hidden_con = self.con_flow.get_init_hidden(hidden_uncon, memorized_v)  # n_memorized_nodes x n_dims

        #self.graph.set_init(node_attention, epoch, batch_i)

        # hidden_uncon: 1 x n_nodes x n_dims_lg
        # hidden_con: n_memorized_nodes x n_dims
        # node_attention: batch_size x n_nodes
        # query_context: batch_size x n_dims_lg
        return hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb

    def flow(self, hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb, epoch, tc=None):
        """ hidden_uncon: 1 x n_nodes x n_dims_lg
            hidden_con: n_memorized_nodes x n_dims
            node_attention: batch_size x n_nodes
        """
        ''' run unconsciousness flow '''
        if self.hparams.simultaneous_uncon_flow:
            # candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
            candidate_edges = self.graph.get_candidate_edges(tc=get(tc, 'graph'))

            # loglog_u: (np.array) n_candidate_edges
            # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
            loglog_u, sampled_edges = self.graph.sample_edges(candidate_edges, self.sampler.edges_logits,
                                                              mode='by_eg',
                                                              max_edges_per_eg=self.hparams.max_edges_per_example,
                                                              use_gumbel_softmax=self.hparams.use_gumbel_softmax,
                                                              tc=get(tc, 'graph'))

            # edges_y: (tf.Tensor) n_sample_edges
            diff_control = self.get_hparam_by_epoch(self.hparams.diff_control_for_sampling, epoch)
            edges_y = self.sampler(None, candidate_edges=candidate_edges, loglog_u=loglog_u,
                                   sampled_edges=sampled_edges, mode='by_eg', diff_control=diff_control,
                                   tc=get(tc, 'model'))

            # selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj] sorted by (eg_idx, vi, vj)
            selected_edges = self.graph.get_selected_edges(sampled_edges, tc=get(tc, 'graph'))

            new_hidden_uncon = self.uncon_flow(hidden_uncon, selected_edges=selected_edges, edges_y=edges_y,
                                               shared_embedding=self.shared_embedding,
                                               tc=get(tc, 'model'))  # 1 x n_nodes x n_dims_lg
        else:
            new_hidden_uncon = hidden_uncon  # 1 x n_nodes x n_dims_lg

        ''' get scanned edges '''
        # attended_nodes: (np.array) n_attended_nodes x 2, (eg_idx, vi) sorted
        attended_nodes = self.graph.get_topk_nodes(node_attention, self.hparams.max_attended_nodes,
                                                   tc=get(tc, 'graph'))  # n_attended_nodes x 2

        # candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
        candidate_edges = self.graph.get_candidate_edges(attended_nodes=attended_nodes,
                                                         tc=get(tc, 'graph'))  # n_candidate_edges x 2

        # loglog_u: (np.array) n_candidate_edges
        # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
        loglog_u, sampled_edges = self.graph.sample_edges(candidate_edges, self.sampler.edges_logits,
                                                          mode='by_vi',
                                                          max_edges_per_vi=self.hparams.max_edges_per_node,
                                                          use_gumbel_softmax=self.hparams.use_gumbel_softmax,
                                                          tc=get(tc, 'graph'))

        # edges_y: (tf.Tensor) n_sample_edges
        diff_control = self.get_hparam_by_epoch(self.hparams.diff_control_for_sampling, epoch)
        edges_y = self.sampler(None, candidate_edges=candidate_edges, loglog_u=loglog_u,
                               sampled_edges=sampled_edges, mode='by_vi', diff_control=diff_control,
                               tc=get(tc, 'model'))

        # scanned_edges: (np.array) n_scanned_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        scanned_edges = self.graph.get_selected_edges(sampled_edges, tc=get(tc, 'graph'))

        ''' add selfloop and backtrace edges '''
        selfloop_edges, backtrace_edges = self.graph.get_selfloop_and_backtrace(attended_nodes,
                                                                                self.hparams.max_backtrace_nodes,
                                                                                tc=get(tc, 'graph'))

        # aug_scanned_edges: n_aug_scanned_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        aug_scanned_edges, new_idx_for_edges_y, rest_idx = self.graph.get_union_edges(scanned_edges,
                                                                                      selfloop_edges,
                                                                                      backtrace_edges,
                                                                                      tc=get(tc, 'graph'))
        edges_y = tf.scatter_nd(new_idx_for_edges_y, edges_y, tf.TensorShape((aug_scanned_edges.shape[0],)))
        rest_y = tf.scatter_nd(rest_idx, tf.ones((rest_idx.shape[0],)), tf.TensorShape((aug_scanned_edges.shape[0],)))
        edges_y = edges_y + rest_y  # n_aug_scanned_edges

        ''' run attention flow (over memorized and scanned nodes) '''
        # new_idx_for_memorized: n_memorized_nodes x 1 or None
        # memorized_and_scanned: n_memorized_and_scanned_nodes x 2, (eg_idx, v) sorted by (eg_idx, v)
        new_idx_for_memorized, n_memorized_and_scanned_nodes, memorized_and_scanned = \
            self.graph.add_nodes_to_memorized(scanned_edges, tc=get(tc, 'graph'))

        # aug_scanned_edges: n_aug_scanned_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        aug_scanned_edges = self.graph.set_index_over_nodes(aug_scanned_edges, memorized_and_scanned, tc=get(tc, 'graph'))

        if not self.hparams.use_ent_emb:
            query_heads = np.stack([np.arange(len(self.heads)).astype('int32'), self.heads], axis=1)
            query_head_idx = self.graph.get_index_from_nodes(query_heads, memorized_and_scanned)
        else:
            query_head_idx = None

        diff_control = self.get_hparam_by_epoch(self.hparams.diff_control_for_transition, epoch)
        trans_attention, new_node_attention = self.att_flow(node_attention,
                                                            scanned_edges=aug_scanned_edges,
                                                            edges_y=edges_y,
                                                            hidden_uncon=new_hidden_uncon,
                                                            hidden_con=hidden_con,
                                                            query_head_emb=query_head_emb,
                                                            query_rel_emb=query_rel_emb,
                                                            query_head_idx=query_head_idx,
                                                            new_idx_for_memorized=new_idx_for_memorized,
                                                            n_memorized_and_scanned_nodes=n_memorized_and_scanned_nodes,
                                                            shared_embedding=self.shared_embedding,
                                                            diff_control=diff_control,
                                                            tc=get(tc, 'model'))  # n_aug_scanned_edges, batch_size x n_nodes

        ''' get seen edges '''
        # seen_nodes: (np.array) n_seen_nodes x 2, (eg_idx, vj) unique and sorted
        seen_nodes = self.graph.get_topk_nodes(new_node_attention, self.hparams.max_seen_nodes,
                                               tc=get(tc, 'graph'))  # n_seen_nodes x 2

        # seen_edges: (np.array) n_seen_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        # seen_idx_for_edges_y: (np.array) n_seen_edges x 1
        seen_edges, seen_idx_for_edges_y = self.graph.get_seen_edges(seen_nodes, aug_scanned_edges, tc=get(tc, 'graph'))

        edges_y = tf.gather_nd(edges_y, seen_idx_for_edges_y)  # n_seen_edges
        trans_attention = tf.gather_nd(trans_attention, seen_idx_for_edges_y)  # n_seen_edges

        ''' run consciousness flow (over memorized and seen nodes) '''
        # new_idx_for_memorized: n_memorized_nodes x 1 or None
        # memorized_and_seen: _memorized_and_seen_nodes x 2, (eg_idx, v) sorted by (eg_idx, v)
        new_idx_for_memorized, n_memorized_and_seen_nodes, memorized_and_seen = \
            self.graph.add_nodes_to_memorized(seen_edges, node_attention=new_node_attention,
                                              backtrace_decay=self.hparams.backtrace_decay,
                                              inplace=True, tc=get(tc, 'graph'))

        if new_idx_for_memorized is not None:
            hidden_con = tf.scatter_nd(new_idx_for_memorized, hidden_con,
                                       tf.TensorShape((n_memorized_and_seen_nodes, self.hparams.n_dims)))  # n_memorized_nodes (new) x n_dims

        # seen_edges: n_seen_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        seen_edges = self.graph.set_index_over_nodes(seen_edges, memorized_and_seen, tc=get(tc, 'graph'))

        if not self.hparams.use_ent_emb:
            query_heads = np.stack([np.arange(len(self.heads)).astype('int32'), self.heads], axis=1)
            query_head_idx = self.graph.get_index_from_nodes(query_heads, self.graph.memorized_nodes)
        else:
            query_head_idx = None

        new_hidden_con = self.con_flow(hidden_con,
                                       seen_edges=seen_edges,
                                       edges_y=edges_y,
                                       trans_attention=trans_attention,
                                       node_attention=new_node_attention,
                                       hidden_uncon=new_hidden_uncon,
                                       memorized_nodes=self.graph.memorized_nodes,
                                       query_head_emb=query_head_emb,
                                       query_rel_emb=query_rel_emb,
                                       query_head_idx=query_head_idx,
                                       shared_embedding=self.shared_embedding,
                                       tc=get(tc, 'model'))  # n_memorized_nodes (new) x n_dims

        ''' do storing work at the end of each step '''
        #self.graph.add_flow(new_node_attention, attended_nodes)

        # new_hidden_uncon: 1 x n_nodes x n_dims_lg,
        # new_hidden_con: n_memorized_nodes x n_dims,
        # new_node_attention: batch_size x n_nodes
        return new_hidden_uncon, new_hidden_con, new_node_attention

    @property
    def regularization_loss(self):
        return tf.add_n(self.shared_embedding.losses)

    @property
    def trainable_variables(self):
        return self.sampler.trainable_variables + \
               self.shared_embedding.trainable_variables + \
               self.query_context.trainable_variables + \
               self.uncon_flow.trainable_variables + \
               self.con_flow.trainable_variables + \
               self.att_flow.trainable_variables


""" run.py """

import config_D as config

parser = argparse.ArgumentParser()
parser = config.get_fb237_config(parser)
default_hparams = parser.parse_args()


def loss_fn(prediction, tails):
    """ predictions: (tf.Tensor) batch_size x n_nodes
        tails: (np.array) batch_size
    """
    pred_idx = tf.stack([tf.range(0, len(tails)), tails], axis=1)  # batch_size x 2
    pred_prob = tf.gather_nd(prediction, pred_idx)  # batch_size
    pred_loss = tf.reduce_mean(- tf.math.log(pred_prob + 1e-20))
    return pred_loss


def calc_metric(heads, relations, prediction, targets, filter_pool):
    hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = 0., 0., 0., 0., 0., 0., 0.

    n_preds = prediction.shape[0]
    for i in range(n_preds):
        head = heads[i]
        rel = relations[i]
        tar = targets[i]
        pred = prediction[i]
        fil = list(filter_pool[(head, rel)] - {tar})

        sorted_idx = np.argsort(-pred)
        mask = np.logical_not(np.isin(sorted_idx, fil))
        sorted_idx = sorted_idx[mask]

        rank = np.where(sorted_idx == tar)[0].item() + 1

        if rank <= 1:
            hit_1 += 1
        if rank <= 3:
            hit_3 += 1
        if rank <= 5:
            hit_5 += 1
        if rank <= 10:
            hit_10 += 1
        mr += rank
        mrr += 1. / rank
        max_r = max(max_r, rank)

    hit_1 /= n_preds
    hit_3 /= n_preds
    hit_5 /= n_preds
    hit_10 /= n_preds
    mr /= n_preds
    mrr /= n_preds

    return hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r


class Trainer(object):
    def __init__(self, model, data_env, hparams):
        self.model = model
        self.data_env = data_env
        self.hparams = hparams
        self.optimizer = keras.optimizers.Adam(learning_rate=hparams.learning_rate)

        self.train_loss = None
        self.train_accuracy = None

    def train_step(self, heads, tails, rels, epoch, batch_i, tc=None):
        n_splits = 1
        heads_li, tails_li, rels_li = [heads], [tails], [rels]
        while True:
            try:
                train_loss = 0.
                accuracy = 0.
                prediction_li = []
                for k in range(n_splits):
                    heads,tails, rels = heads_li[k], tails_li[k], rels_li[k]

                    with tf.GradientTape() as tape:
                        predictions = []
                        for i in range(self.hparams.n_rollouts_for_train):
                            hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb = \
                                self.model.initialize(heads, rels, epoch, batch_i, tc=tc)

                            for step in range(1, self.hparams.max_steps + 1):
                                hidden_uncon, hidden_con, node_attention = \
                                    self.model.flow(hidden_uncon, hidden_con, node_attention,
                                                    query_head_emb, query_rel_emb, epoch, tc=tc)
                            predictions.append(node_attention)

                            #pred_tails = tf.argmax(node_attention, axis=1)
                            #self.model.graph.add_inputs(np.stack([heads, rels, tails, pred_tails], axis=1), epoch, batch_i)

                        prediction = tf.reduce_mean(tf.stack(predictions, axis=2), axis=2)  # batch_size x n_nodes

                        pred_loss = loss_fn(prediction, tails)
                        reg_loss = self.model.regularization_loss
                        loss = pred_loss + reg_loss

                    if tc is not None:
                        t0 = time.time()
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    if tc is not None:
                        tc['grad']['comp'] += time.time() - t0

                    if tc is not None:
                        t0 = time.time()

                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    if tc is not None:
                        tc['grad']['apply'] += time.time() - t0

                    train_loss += loss
                    accuracy += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1), tails), tf.float32))
                    prediction_li.append(prediction.numpy())

                train_loss /= n_splits
                accuracy /= n_splits

                decay = self.hparams.moving_mean_decay
                self.train_loss = train_loss if self.train_loss is None else self.train_loss * decay + train_loss * (1 - decay)
                self.train_accuracy = accuracy if self.train_accuracy is None else self.train_accuracy * decay + accuracy * (1 - decay)

                train_metric = None
                if self.hparams.print_train_metric:
                    train_metric = calc_metric(np.concatenate(heads_li, axis=0),
                                               np.concatenate(rels_li, axis=0),
                                               np.concatenate(prediction_li, axis=0),
                                               np.concatenate(tails_li, axis=0),
                                               self.data_env.filter_pool)

                return train_loss.numpy(), accuracy.numpy(), self.train_loss.numpy(), self.train_accuracy.numpy(), train_metric

            except (tf.errors.InternalError, tf.errors.ResourceExhaustedError, SystemError):
                print('Meet `tf.errors.InternalError` or `tf.errors.ResourceExhaustedError` or `SystemError`')
                n_splits += 1
                print('split into %d batches' % n_splits)
                heads_li = np.array_split(heads, n_splits, axis=0)
                tails_li = np.array_split(tails, n_splits, axis=0)
                rels_li = np.array_split(rels, n_splits, axis=0)


class Evaluator(object):
    def __init__(self, model, data_env, hparams):
        self.model = model
        self.data_env = data_env
        self.hparams = hparams

        self.heads = []
        self.relations = []
        self.prediction = []
        self.targets = []

        self.eval_loss = None
        self.eval_accuracy = None

    def eval_step(self, heads, tails, rels, epoch, batch_i):
        n_splits = 1
        heads_li, tails_li, rels_li = [heads], [tails], [rels]
        while True:
            try:
                eval_loss = 0.
                accuracy = 0.
                prediction_li = []
                for k in range(n_splits):
                    heads, tails, rels = heads_li[k], tails_li[k], rels_li[k]

                    predictions = []
                    for i in range(self.hparams.n_rollouts_for_eval):
                        hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb = \
                            self.model.initialize(heads, rels, epoch, batch_i)

                        for step in range(1, self.hparams.max_steps + 1):
                            hidden_uncon, hidden_con, node_attention = \
                                self.model.flow(hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb, epoch)
                        predictions.append(node_attention)

                        #pred_tails = tf.argmax(node_attention, axis=1)
                        #self.model.graph.add_inputs(np.stack([heads, rels, tails, pred_tails], axis=1), epoch, batch_i)

                    prediction = tf.reduce_mean(tf.stack(predictions, axis=2), axis=2)  # batch_size x n_nodes

                    self.heads.append(heads)
                    self.relations.append(rels)
                    self.prediction.append(prediction.numpy())
                    self.targets.append(tails)

                    pred_loss = loss_fn(prediction, tails)
                    reg_loss = self.model.regularization_loss
                    loss = pred_loss + reg_loss

                    eval_loss += loss
                    accuracy += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1), tails), tf.float32))
                    prediction_li.append(prediction.numpy())

                eval_loss /= n_splits
                accuracy /= n_splits

                decay = self.hparams.moving_mean_decay
                self.eval_loss = eval_loss if self.eval_loss is None else self.eval_loss * decay + eval_loss * (1 - decay)
                self.eval_accuracy = accuracy if self.eval_accuracy is None else self.eval_accuracy * decay + accuracy * (1 - decay)

                eval_metric = None
                if self.hparams.print_eval_metric:
                    eval_metric = calc_metric(np.concatenate(heads_li, axis=0),
                                              np.concatenate(rels_li, axis=0),
                                              np.concatenate(prediction_li, axis=0),
                                              np.concatenate(tails_li, axis=0),
                                              self.data_env.filter_pool)

                return eval_loss.numpy(), accuracy.numpy(), self.eval_loss.numpy(), self.eval_accuracy.numpy(), eval_metric

            except (tf.errors.InternalError, tf.errors.ResourceExhaustedError, SystemError):
                print('Meet `tf.errors.InternalError` or `tf.errors.ResourceExhaustedError` or `SystemError`')
                n_splits += 1
                print('split into %d batches' % n_splits)
                heads_li = np.array_split(heads, n_splits, axis=0)
                tails_li = np.array_split(tails, n_splits, axis=0)
                rels_li = np.array_split(rels, n_splits, axis=0)

    def reset_metric(self):
        self.heads = []
        self.relations = []
        self.prediction = []
        self.targets = []

    def metric_result(self):
        heads = np.concatenate(self.heads, axis=0)
        relations = np.concatenate(self.relations, axis=0)
        prediction = np.concatenate(self.prediction, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        return calc_metric(heads, relations, prediction, targets, self.data_env.filter_pool)


def reset_time_cost(hparams):
    if hparams.timer:
        return {'model': defaultdict(float), 'graph': defaultdict(float), 'grad': defaultdict(float)}
    else:
        return None


def str_time_cost(tc):
    if tc is not None:
        model_tc = ', '.join('m.{} {:3f}'.format(k, v) for k, v in tc['model'].items())
        graph_tc = ', '.join('g.{} {:3f}'.format(k, v) for k, v in tc['graph'].items())
        grad_tc = ', '.join('d.{} {:3f}'.format(k, v) for k, v in tc['grad'].items())
        return model_tc + ', ' + graph_tc + ', ' + grad_tc
    else:
        return ''


def run(dataset, hparams):
    data_env = DataEnv(dataset, hparams)
    model = Model(data_env.graph, hparams)
    trainer = Trainer(model, data_env, hparams)
    valid_evaluator = Evaluator(model, data_env, hparams)
    test_evaluator = Evaluator(model, data_env, hparams)

    train_batcher = data_env.get_train_batcher()
    valid_batcher = data_env.get_valid_batcher()
    test_batcher = data_env.get_test_batcher()

    for epoch in range(1, hparams.max_epochs + 1):
        batch_i = 1

        valid_batch_gen = valid_batcher(hparams.batch_size)
        test_batch_gen = test_batcher(hparams.batch_size)
        for train_batch, batch_size in train_batcher(hparams.batch_size):
            t0 = time.time()
            time_cost = reset_time_cost(hparams)

            heads, tails, rels = train_batch[:, 0], train_batch[:, 1], train_batch[:, 2]
            cur_train_loss, cur_accuracy, train_loss, accuracy, train_metric = trainer.train_step(
                heads, tails, rels, epoch, batch_i, tc=time_cost)

            dt = time.time() - t0

            if hparams.print_train and batch_i % hparams.print_train_freq == 0:
                if hparams.print_train_metric:
                    hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = train_metric
                    print('[TRAIN] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} {} | '
                          'hit_1: {:.4f} | hit_3: {:.4f} | hit_5: {:.4f} | hit_10: {:.4f} | '
                          'mr: {:.1f} | mmr: {:.4f} | max_r: {:.1f}'.format(
                        epoch, batch_i, train_loss, cur_train_loss, accuracy, cur_accuracy,
                        dt, str_time_cost(time_cost), hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))
                else:
                    print('[TRAIN] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} {}'.format(
                        epoch, batch_i, train_loss, cur_train_loss, accuracy, cur_accuracy, dt, str_time_cost(time_cost)))

            if hparams.print_eval and batch_i % hparams.print_eval_freq == 0:
                try:
                    valid_batch, batch_size = next(valid_batch_gen)
                except StopIteration:
                    valid_batch_gen = valid_batcher(hparams.batch_size)
                    valid_batch, batch_size = next(valid_batch_gen)
                t0 = time.time()

                heads, tails, rels = valid_batch[:, 0], valid_batch[:, 1], valid_batch[:, 2]
                cur_eval_loss, cur_accuracy, eval_loss, accuracy, eval_metric = valid_evaluator.eval_step(
                    heads, tails, rels, epoch, batch_i)

                dt = time.time() - t0

                if hparams.print_eval_metric:
                    hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = eval_metric
                    print('[VALID] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} | '
                          'hit_1: {:.4f} | hit_3: {:.4f} | hit_5: {:.4f} | hit_10: {:.4f} | '
                          'mr: {:.1f} | mmr: {:.4f} | max_r: {:.1f} * * * * *'.format(
                        epoch, batch_i, eval_loss, cur_eval_loss, accuracy, cur_accuracy, dt,
                        hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))
                else:
                    print('[VALID] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} * * * * *'.format(
                        epoch, batch_i, eval_loss, cur_eval_loss, accuracy, cur_accuracy, dt))

                try:
                    test_batch, batch_size = next(test_batch_gen)
                except StopIteration:
                    test_batch_gen = test_batcher(hparams.batch_size)
                    test_batch, batch_size = next(test_batch_gen)
                t0 = time.time()

                heads, tails, rels = test_batch[:, 0], test_batch[:, 1], test_batch[:, 2]
                cur_eval_loss, cur_accuracy, eval_loss, accuracy, eval_metric = test_evaluator.eval_step(
                    heads, tails, rels, epoch, batch_i)

                dt = time.time() - t0

                if hparams.print_eval_metric:
                    hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = eval_metric
                    print('[TEST] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} | '
                          'hit_1: {:.4f} | hit_3: {:.4f} | hit_5: {:.4f} | hit_10: {:.4f} | '
                          'mr: {:.1f} | mmr: {:.4f} | max_r: {:.1f} * * * * *'.format(
                        epoch, batch_i, eval_loss, cur_eval_loss, accuracy, cur_accuracy, dt,
                        hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))
                else:
                    print('[TEST] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} * * * * *'.format(
                        epoch, batch_i, eval_loss, cur_eval_loss, accuracy, cur_accuracy, dt))

            batch_i += 1

        valid_evaluator.reset_metric()
        batch_i = 1
        for valid_batch, batch_size in valid_batcher(hparams.batch_size):
            t0 = time.time()

            heads, tails, rels = valid_batch[:, 0], valid_batch[:, 1], valid_batch[:, 2]
            cur_eval_loss, cur_accuracy, eval_loss, accuracy, eval_metric = valid_evaluator.eval_step(
                heads, tails, rels, epoch, batch_i)

            dt = time.time() - t0

            #if hparams.print_eval:
            #    if hparams.print_eval_metric:
            #        hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = eval_metric
            #        print('[FULL VALID] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} | '
            #              'hit_1: {:.4f} | hit_3: {:.4f} | hit_5: {:.4f} | hit_10: {:.4f} | '
            #              'mr: {:.1f} | mmr: {:.4f} | max_r: {:.1f} * * * * *'.format(
            #            epoch, batch_i, eval_loss, cur_eval_loss, accuracy, cur_accuracy, dt,
            #            hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))
            #    else:
            #        print('[FULL VALID] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} * * * * *'.format(
            #            epoch, batch_i, eval_loss, cur_eval_loss, accuracy, cur_accuracy, dt))
            batch_i += 1

        hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = valid_evaluator.metric_result()
        print('[REPORT VALID] {:d} | hit_1: {:.6f} | hit_3: {:.6f} | hit_5: {:.6f} | hit_10: {:.6f} | '
              'mr: {:.1f} | mmr: {:.6f} | max_r: {:.1f} * * * * *'.format(epoch, hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))

        test_evaluator.reset_metric()
        batch_i = 1
        for test_batch, batch_size in test_batcher(hparams.batch_size):
            t0 = time.time()

            heads, tails, rels = test_batch[:, 0], test_batch[:, 1], test_batch[:, 2]
            cur_eval_loss, cur_accuracy, eval_loss, accuracy, eval_metric = test_evaluator.eval_step(
                heads, tails, rels, epoch, batch_i)

            dt = time.time() - t0

            # if hparams.print_eval:
            #     if hparams.print_eval_metric:
            #         hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = eval_metric
            #         print('[FULL TEST] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} | '
            #               'hit_1: {:.4f} | hit_3: {:.4f} | hit_5: {:.4f} | hit_10: {:.4f} | '
            #               'mr: {:.1f} | mmr: {:.4f} | max_r: {:.1f} * * * * *'.format(
            #             epoch, batch_i, eval_loss, cur_eval_loss, accuracy, cur_accuracy, dt,
            #             hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))
            #     else:
            #         print('[FULL TEST] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} * * * * *'.format(
            #             epoch, batch_i, eval_loss, cur_eval_loss, accuracy, cur_accuracy, dt))
            batch_i += 1

        hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = test_evaluator.metric_result()
        print('[REPORT TEST] {:d} | hit_1: {:.6f} | hit_3: {:.6f} | hit_5: {:.6f} | hit_10: {:.6f} | '
              'mr: {:.1f} | mmr: {:.6f} | max_r: {:.1f} * * * * *'.format(epoch, hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))

    print('DONE')

if __name__ == '__main__':
    hparams = copy.deepcopy(default_hparams)
    print(hparams)

    dataset = getattr(datasets, hparams.dataset)()
    run(dataset, hparams)
