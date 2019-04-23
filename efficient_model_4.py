from collections import defaultdict
import time

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras

from utils import get

class F(keras.layers.Layer):
    def __init__(self, interact, n_dims, use_bias=True, activation=None, output_weight=False, output_bias=False, name=None):
        super(F, self).__init__(name=name)
        self.interact = interact
        self.n_dims = n_dims
        self.use_bias = use_bias
        self.activation = activation
        self.output_weight = output_weight
        self.output_bias = output_bias

    def build(self, input_shape):
        n_ws = len(self.interact)
        self.ws = self.add_weight(shape=(n_ws, self.n_dims), initializer=keras.initializers.VarianceScaling(), name='ws')
        if self.use_bias:
            self.b = self.add_weight(shape=(self.n_dims,), initializer=keras.initializers.zeros(), name='b')
        if self.output_weight:
            self.out_w = self.add_weight(shape=(self.n_dims,), initializer=keras.initializers.VarianceScaling(), name='out_w')
        if self.output_bias:
            self.out_b = self.add_weight(shape=(self.n_dims,), initializer=keras.initializers.zeros(), name='out_b')

    def call(self, inputs, training=None):
        """ inputs[i]: bs x ... x n_dims
        """
        xs = []
        for idxs in self.interact:
            x = 1.
            for idx in idxs:
                x = x * inputs[idx]
            xs.append(x)  # x: bs x ... x n_dims
        xs = tf.stack(xs, axis=-2)  # bs x ... x n_ws x n_dims
        outputs = tf.reduce_sum(xs * self.ws, axis=-2)  # bs x ... x n_dims
        if self.use_bias:
            outputs = outputs + self.b
        if self.activation:
            outputs = self.activation(outputs)
        if self.output_weight:
            outputs = outputs * self.out_w
        if self.output_bias:
            outputs = outputs + self.out_b
        return outputs


class G(keras.layers.Layer):
    def __init__(self, n_dims, use_bias=True, activation=None, output_weight=False, output_bias=False, residual=True, name=None):
        super(G, self).__init__(name=name)
        self.n_dims = n_dims
        self.use_bias = use_bias
        self.activation = activation
        self.output_bias = output_bias
        self.output_weight = output_weight
        self.residual = residual

    def build(self, input_shape):
        self.dense = keras.layers.Dense(self.n_dims, activation=self.activation, use_bias=self.use_bias, name='den')
        if self.output_weight:
            self.dense_out = keras.layers.Dense(self.n_dims, use_bias=self.output_bias, name='out')

    def call(self, inputs, training=None):
        outputs = self.dense(inputs)
        if self.output_weight:
            outputs = self.dense_out(outputs)
        if self.residual:
            outputs = inputs + outputs
        return outputs


class ReducedGRU(keras.layers.Layer):
    def __init__(self, n_dims, name=None):
        super(ReducedGRU, self).__init__(n_dims, name=name)
        self.n_dims = n_dims

    def build(self, input_shape):
        self.z_gate = keras.layers.Dense(self.n_dims, activation=tf.math.sigmoid, name='z_gate')

    def call(self, inputs, training=None):
        prev_hidden, update = inputs
        z = self.z_gate(tf.concat([prev_hidden, update], -1))
        new_hidden = (1 - z) * prev_hidden + z * update
        return new_hidden


class Node2Edge(keras.layers.Layer):
    def call(self, inputs, selected_edges=None, return_vi=True, return_vj=True, training=None):
        """ inputs (hidden): batch_size x n_nodes x n_dims
            selected_edges: n_selected_edges x 6 (or 8) ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
        """
        assert selected_edges is not None
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


class Node2Edge_v2(keras.layers.Layer):
    def call(self, inputs, selected_edges=None, return_vi=True, return_vj=True, training=None):
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


class Aggregate(keras.layers.Layer):
    def call(self, inputs, selected_edges=None, output_shape=None, at='vj', aggr_op='mean', training=None):
        """ inputs (edge_vec): n_seleted_edges x ...
            selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
            output_shape: (batch_size=1, n_nodes, ...)
        """
        assert selected_edges is not None
        assert output_shape is not None
        edge_vec = inputs
        if at == 'vi':
            idx_vi = selected_edges[:, 4]  # n_selected_edges
            aggr_op = tf.math.segment_mean if aggr_op == 'mean' else \
                tf.math.segment_sum if aggr_op == 'sum' else \
                tf.math.segment_max if aggr_op == 'max' else None
            edge_vec_aggr = aggr_op(edge_vec, idx_vi)  # (max_idx_vi+1) x ...
            idx_and_vi = tf.stack([selected_edges[:, 0], selected_edges[:, 1]], axis=1)  # n_selected_edges x 2
            idx_and_vi = tf.cast(tf.math.segment_max(idx_and_vi, idx_vi), tf.int32)  # (max_id_vi+1) x 2
            edge_vec_aggr = tf.scatter_nd(idx_and_vi, edge_vec_aggr, output_shape)  # batch_size x n_nodes x ...
        elif at == 'vj':
            idx_vj = selected_edges[:, 5]  # n_selected_edges
            max_idx_vj = tf.reduce_max(idx_vj)
            aggr_op = tf.math.unsorted_segment_mean if aggr_op == 'mean' else \
                tf.math.unsorted_segment_sum if aggr_op == 'sum' else \
                tf.math.unsorted_segment_max if aggr_op == 'max' else None
            edge_vec_aggr = aggr_op(edge_vec, idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x ...
            idx_and_vj = tf.stack([selected_edges[:, 0], selected_edges[:, 2]], axis=1)  # n_selected_edges x 2
            idx_and_vj = tf.cast(tf.math.unsorted_segment_max(idx_and_vj, idx_vj, max_idx_vj + 1), tf.int32)  # (max_idx_vj+1) x 2
            edge_vec_aggr = tf.scatter_nd(idx_and_vj, edge_vec_aggr, output_shape)  # batch_size x n_nodes x ...
        else:
            raise ValueError('Invalid `at`')
        return edge_vec_aggr


class Aggregate_v2(keras.layers.Layer):
    def call(self, inputs, selected_edges=None, output_shape=None, at='vj', aggr_op='mean', training=None):
        """ inputs (edge_vec): n_seleted_edges x ...
            selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
            output_shape: (n_visited_nodes, ...)
        """
        assert selected_edges is not None
        assert output_shape is not None
        edge_vec = inputs
        if at == 'vi':
            idx_vi = selected_edges[:, 4]  # n_selected_edges
            aggr_op = tf.math.segment_mean if aggr_op == 'mean' else \
                tf.math.segment_sum if aggr_op == 'sum' else \
                tf.math.segment_max if aggr_op == 'max' else None
            edge_vec_aggr = aggr_op(edge_vec, idx_vi)  # (max_idx_vi+1) x ...
            new_idx_e2vi = selected_edges[:, 6]  # n_selected_edges
            reduced_idx_e2vi = tf.cast(tf.math.segment_max(new_idx_e2vi, idx_vi), tf.int32)  # (max_id_vi+1)
            reduced_idx_e2vi = tf.expand_dims(reduced_idx_e2vi, 1)  # (max_id_vi+1) x 1
            edge_vec_aggr = tf.scatter_nd(reduced_idx_e2vi, edge_vec_aggr, output_shape)  # n_visited_nodes x ...
        elif at == 'vj':
            idx_vj = selected_edges[:, 5]  # n_selected_edges
            max_idx_vj = tf.reduce_max(idx_vj)
            aggr_op = tf.math.unsorted_segment_mean if aggr_op == 'mean' else \
                tf.math.unsorted_segment_sum if aggr_op == 'sum' else \
                tf.math.unsorted_segment_max if aggr_op == 'max' else None
            edge_vec_aggr = aggr_op(edge_vec, idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x ...
            new_idx_e2vj = selected_edges[:, 7]  # n_selected_edges
            reduced_idx_e2vj = tf.cast(tf.math.unsorted_segment_max(new_idx_e2vj, idx_vj, max_idx_vj + 1), tf.int32)  # (max_idx_vj+1)
            reduced_idx_e2vj = tf.expand_dims(reduced_idx_e2vj, 1)  # (max_idx_vj+1) x 1
            edge_vec_aggr = tf.scatter_nd(reduced_idx_e2vj, edge_vec_aggr, output_shape)  # n_visited_nodes x ...
        else:
            raise ValueError('Invalid `at`')
        return edge_vec_aggr


def sparse_softmax(logits, segment_ids, sort=True):
    if sort:
        logits_max = tf.math.segment_max(logits, segment_ids)
        logits_max = tf.gather(logits_max, segment_ids)
        logits_exp = tf.math.exp(logits - logits_max)
        logits_expsum = tf.math.segment_sum(logits_exp, segment_ids)
        logits_expsum = tf.gather(logits_expsum, segment_ids)
        logits_norm = logits_exp / logits_expsum
    else:
        num_segments = tf.reduce_max(segment_ids) + 1
        logits_max = tf.math.unsorted_segment_max(logits, segment_ids, num_segments)
        logits_max = tf.gather(logits_max, segment_ids)
        logits_exp = tf.math.exp(logits - logits_max)
        logits_expsum = tf.math.unsorted_segment_sum(logits_exp, segment_ids, num_segments)
        logits_expsum = tf.gather(logits_expsum, segment_ids)
        logits_norm = logits_exp / logits_expsum
    return logits_norm


class NeighborSoftmax(keras.layers.Layer):
    def call(self, inputs, selected_edges=None, at='vi', training=None):
        """ inputs (edge_vec): n_seleted_edges x ...
            selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
        """
        assert selected_edges is not None
        edge_vec = inputs
        if at == 'vi':
            idx_vi = selected_edges[:, 4]  # n_selected_edges
            edge_vec_norm = sparse_softmax(edge_vec, idx_vi)  # n_selected_edges x ...
        elif at == 'vj':
            idx_vj = selected_edges[:, 5]  # n_selected_edges
            edge_vec_norm = sparse_softmax(edge_vec, idx_vj, sort=False)  # n_selected_edges x ...
        else:
            raise ValueError('Invalid `at`')
        return edge_vec_norm


class Sampler(keras.Model):
    def __init__(self, graph):
        super(Sampler, self).__init__(name='sampler')
        with tf.name_scope(self.name):
            with tf.device('/cpu:0'):
                self.edges_logits = self.add_weight(shape=(graph.n_full_edges,),
                                                    initializer=keras.initializers.constant(self._initialize(graph)),
                                                    name='edges_logits')  # n_full_edges

    def _initialize(self, graph):
        logits_init = np.zeros((graph.n_full_edges,), np.float32)
        for e_id, vi, rel, vj in graph.full_edges:
            logits_init[e_id] = np.log((1. / graph.count(vi)) *
                                       (1. / graph.count((vi, rel))) *
                                       (1. / graph.count((vi, rel, vj))))
        return logits_init

    def call(self, _, candidate_edges=None, loglog_u=None, sampled_edges=None, mode=None, training=None, tc=None):
        """ inputs: None
            candidate_edges: (pd.DataFram) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
            loglog_u: (np.array) n_candidate_edges
            sampled_edges: (pd.DataFrame) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
        """
        assert candidate_edges is not None
        assert loglog_u is not None
        assert sampled_edges is not None
        assert mode is not None
        if tc is not None:
            t0 = time.time()

        if mode == 'by_eg':
            segment_ids = candidate_edges['eg_idx'].values  # n_candidate_edges

        elif mode == 'by_vi':
            idx_and_vi = candidate_edges[['eg_idx', 'vi']].values
            _, segment_ids = np.unique(idx_and_vi, axis=0, return_inverse=True)
            segment_ids = np.array(segment_ids, dtype='int32')  # n_candidate_edges

        else:
            raise ValueError('Invalid `mode`')

        edge_id = candidate_edges['edge_id'].values
        logits = tf.gather(self.edges_logits, edge_id)  # n_candidate_edges
        ca_idx = sampled_edges['ca_idx'].values  # n_sampled_edges

        edges_y = self._gumbel_softmax(logits, loglog_u, segment_ids, ca_idx)

        if tc is not None:
            tc['s.call'] += time.time() - t0
        return edges_y  # n_sampled_edges

    def _gumbel_softmax(self, logits, loglog_u, segment_ids, ca_idx, temperature=1., hard=True):
        y = logits + loglog_u  # n_candidate_edges
        y = sparse_softmax(y / temperature, segment_ids)  # n_candidate_edges
        y = tf.gather(y, ca_idx)  # n_sampled_edges
        if hard:
            y_hard = tf.ones_like(y)
            y = tf.stop_gradient(y_hard - y) + y
        return y  # n_sampled_edges


class UnconsciousnessFlow(keras.Model):
    def __init__(self, n_entities, n_relations, n_dims_lg, ent_emb_l2, rel_emb_l2):
        """ n_entities: including virtual nodes
            n_relations: including 'virtual' but not 'selfloop' and 'backtrace'
            n_dims_lg: uncon flow uses large dimensions
        """
        super(UnconsciousnessFlow, self).__init__(name='uncon_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims_lg

        self.entity_embedding = keras.layers.Embedding(n_entities, self.n_dims,
                                                       embeddings_regularizer=keras.regularizers.l2(ent_emb_l2),
                                                       name='entities')
        self.relation_embedding = keras.layers.Embedding(n_relations, self.n_dims,
                                                         embeddings_regularizer=keras.regularizers.l2(rel_emb_l2),
                                                         name='relations')

        # f(message_aggr, hidden_prev, ent_emb)
        self.f_hidden = F([[0], [0, 1], [0, 2], [1], [2], [1, 2]], self.n_dims,
                          activation=tf.tanh, name='f_hidden')
        self.g_hidden = G(self.n_dims, activation=tf.tanh, name='g_hidden')

        # f(hidden_vi, rel_emb, hidden_vj) (vi -> vj: non-symmetric)
        self.f_message = F([[0], [0, 1], [0, 1, 2]], self.n_dims,
                           activation=tf.tanh, name='f_msg')
        self.g_message = G(self.n_dims, activation=tf.tanh, name='g_msg')

        self.nodes_to_edges = Node2Edge()

        self.aggregate = Aggregate()

        self.gru = ReducedGRU(self.n_dims, name='gru')

    def call(self, inputs, selected_edges=None, edges_y=None, training=None, tc=None):
        """ inputs (hidden): 1 x n_nodes x n_dims_lg
            selected_edges: n_selected_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
            edges_y: n_selected_edges

            Here: batch_size = 1
        """
        assert selected_edges is not None
        assert edges_y is not None
        if tc is not None:
            t0 = time.time()

        # compute unconscious messages
        hidden = inputs
        hidden_vi, hidden_vj = self.nodes_to_edges(hidden, selected_edges=selected_edges)  # n_selected_edges x n_dims_lg
        rel_idx = selected_edges[:, 3]  # n_selected_edges
        rel_emb = self.relation_embedding(rel_idx)  # n_selected_edges x n_dims_lg
        message = self.f_message((hidden_vi, rel_emb, hidden_vj))  # n_selected_edges x n_dims_lg
        message = self.g_message(message)  # n_selected_edges x n_dims_lg
        message = tf.expand_dims(edges_y, 1) * message  # n_selected_edges x n_dims_lg

        # aggregate unconscious messages
        message_aggr = self.aggregate(message, selected_edges=selected_edges,
                                      output_shape=(1, self.n_nodes, self.n_dims))  # 1 x n_nodes x n_dims_lg

        # update unconscious states
        ent_idx = tf.expand_dims(tf.range(0, self.n_nodes), axis=0)  # 1 x n_nodes
        ent_emb = self.entity_embedding(ent_idx)  # 1 x n_nodes x n_dims_lg
        update = self.f_hidden((message_aggr, hidden, ent_emb))  # 1 x n_nodes x n_dims_lg
        update = self.g_hidden(update)  # 1 x n_nodes x n_dims_lg
        hidden = self.gru((hidden, update))  # 1 x n_nodes x n_dims_lg

        if tc is not None:
            tc['u.call'] += time.time() - t0
        return hidden  # 1 x n_nodes x n_dims_lg

    def get_init_hidden(self, hidden_past=None, tc=None):
        """ hidden_past: 1 x n_nodes x n_dims_lg
        """
        if tc is not None:
            t0 = time.time()

        with tf.name_scope(self.name):
            ent_idx = tf.expand_dims(tf.range(0, self.n_nodes), axis=0)  # 1 x n_nodes
            ent_emb = self.entity_embedding(ent_idx)  # 1 x n_nodes x n_dims_lg
            message_aggr = tf.zeros((1, self.n_nodes, self.n_dims))  # 1 x n_nodes x n_dims_lg
            hidden_prev = tf.zeros((1, self.n_nodes, self.n_dims)) if hidden_past is None else hidden_past  # 1 x n_nodes x n_dims_lg
            update = self.f_hidden((message_aggr, hidden_prev, ent_emb))  # 1 x n_nodes x n_dims_lg
            update = self.g_hidden(update)  # 1 x n_nodes x n_dims_lg
            hidden_init = update if hidden_past is None else self.gru((hidden_past, update))  # 1 x n_nodes x n_dims_lg

        if tc is not None:
            tc['u.init'] += time.time() - t0
        return hidden_init  # 1 x n_nodes x n_dims_lg


class ConsciousnessFlow(keras.Model):
    def __init__(self, n_entities, n_relations, n_dims, ent_emb_l2, rel_emb_l2):
        """ n_entities: including virtual nodes
            n_relations: including 'virtual', 'selfloop' and 'backtrace'
        """
        super(ConsciousnessFlow, self).__init__(name='con_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims

        self.entity_embedding = keras.layers.Embedding(n_entities, self.n_dims,
                                                       embeddings_regularizer=keras.regularizers.l2(ent_emb_l2),
                                                       name='entitys')
        self.relation_embedding = keras.layers.Embedding(n_relations, self.n_dims,
                                                         embeddings_regularizer=keras.regularizers.l2(rel_emb_l2),
                                                         name='relations')

        # f(head_emb, rel_emb)
        self.f_query = F([[0], [1], [0,1]], self.n_dims,
                         activation=tf.tanh, name='f_query')
        self.g_query = G(self.n_dims, activation=tf.tanh, name='g_query')

        # f(message, hidden_uncon, hidden, ent_emb)
        self.f_hidden = F([[0], [0, 2], [0, 3], [1], [1, 2], [1, 3], [2], [3], [2, 3]], self.n_dims,
                          activation=tf.tanh, name='f_hidden')
        self.g_hidden = G(self.n_dims, activation=tf.tanh, name='g_hidden')
        self.proj = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj')

        # f(hidden_vi, rel_emb, hidden_vj)
        self.f_message = F([[0], [0, 1], [0, 1, 2]], self.n_dims,
                           activation=tf.tanh, name='f_msg')
        self.g_message = G(self.n_dims, activation=tf.tanh, name='g_msg')

        # f(trans_attention, message)
        self.f_attended_message = F([[0, 1]], self.n_dims, use_bias=False, name='f_att_msg')

        self.nodes_to_edges_v2 = Node2Edge_v2()

        self.aggregate_v2 = Aggregate_v2()

        self.gru = ReducedGRU(self.n_dims, name='gru')

    def call(self, inputs, seen_edges=None, edges_y=None, trans_attention=None, node_attention=None,
             hidden_uncon=None, memorized_nodes=None, training=None, tc=None):
        """ inputs (hidden): n_memorized_nodes x n_dims
            seen_edges: n_seen_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by (idx, vi, vj)
                (1) including selfloop edges and backtrace edges
                (2) batch_size >= 1
            edges_y: n_seen_edges ( sorted according to seen_edges )
            trans_attention: n_seen_edges ( sorted according to seen_edges )
            node_attention: batch_size x n_nodes
            hidden_uncon: 1 x n_nodes x n_dims_lg
            memorized_nodes: n_memorized_nodes x 2, (eg_idx, v)
        """
        assert seen_edges is not None
        assert edges_y is not None
        assert trans_attention is not None
        assert node_attention is not None
        assert hidden_uncon is not None
        if tc is not None:
            t0 = time.time()

        # compute conscious messages
        hidden = inputs
        hidden_vi, hidden_vj = self.nodes_to_edges_v2(hidden, seen_edges)  # n_seen_edges x n_dims
        rel_idx = seen_edges[:, 3]  # n_seen_edges
        rel_emb = self.relation_embedding(rel_idx)  # n_seen_edges x n_dims
        message = self.f_message((hidden_vi, rel_emb, hidden_vj))  # n_seen_edges x n_dims
        message = self.g_message(message)  # n_seen_edges x n_dims
        message = tf.expand_dims(edges_y, 1) * message

        # attend conscious messages
        message = self.f_attended_message((tf.expand_dims(trans_attention, 1), message))  # n_seen_edges x n_dims

        # aggregate conscious messages
        n_memorized_nodes = tf.shape(hidden)[0]
        message_aggr = self.aggregate_v2(message, selected_edges=seen_edges,
                                         output_shape=(n_memorized_nodes, self.n_dims))  # n_memorized_nodes x n_dims

        # update conscious messages
        idx, v = memorized_nodes[:, 0], memorized_nodes[:, 1]  # n_memorized_nodes, n_memorized_nodes
        v_emb = self.entity_embedding(v)  # n_memorized_nodes x n_dims
        hidden_uncon = tf.squeeze(hidden_uncon, axis=0)  # n_nodes x n_dims_lg
        hidden_uncon = self.proj(hidden_uncon)  # n_nodes x n_dims
        hidden_uncon = tf.gather(hidden_uncon, v)  # n_memorized_nodes x n_dims
        update = self.f_hidden((message_aggr, hidden_uncon, hidden, v_emb))  # n_memorized_nodes x n_dims
        update = self.g_hidden(update)
        hidden = self.gru((hidden, update))

        if tc is not None:
            tc['c.call'] += time.time() - t0
        return hidden  # n_memorized_nodes x n_dims

    def get_query_context(self, heads, rels, tc=None):
        """ heads: batch_size
            rels: batch_size
        """
        if tc is not None:
            t0 = time.time()

        with tf.name_scope(self.name):
            head_emb = self.entity_embedding(heads)  # batch_size x n_dims
            rel_emb = self.relation_embedding(rels)  # batch_size x n_dims
            query_context = self.f_query((head_emb, rel_emb))
            query_context = self.g_query(query_context)

        if tc is not None:
            tc['c.query'] += time.time() - t0
        return query_context  # batch_size x n_dims

    def get_init_hidden(self, query_context, hidden_uncon, memorized_v, tc=None):
        """ query_context: batch_size x n_dims
            hidden_uncon: 1 x n_nodes x n_dims_lg
            memorized_v: n_memorized_nodes (=batch_size) x 2, (eg_idx, v)
        """
        if tc is not None:
            t0 = time.time()

        with tf.name_scope(self.name):
            eg_idx, v = memorized_v[:, 0], memorized_v[:, 1]  # n_memorized_nodes, n_memorized_nodes
            v_emb = self.entity_embedding(v)  # n_memorized_nodes x n_dims
            message = tf.gather(query_context, eg_idx)  # n_memorized_nodes x n_dims
            hidden_uncon = tf.squeeze(hidden_uncon, axis=0)  # n_nodes x n_dims_lg
            hidden_uncon = self.proj(hidden_uncon)  # n_nodes x n_dims
            hidden_uncon = tf.gather(hidden_uncon, v)  # n_memorized_nodes x n_dims
            hidden_prev = tf.zeros_like(v_emb)  # n_memorized_nodes x n_dims
            hidden_init = self.f_hidden((message, hidden_uncon, hidden_prev, v_emb))  # n_memorized_nodes x n_dims
            hidden_init = self.g_hidden(hidden_init)  # n_memorized_nodes x n_dims

        if tc is not None:
            tc['c.init'] += time.time() - t0
        return hidden_init  # n_memorized_nodes x n_dims


class AttentionFlow(keras.Model):
    def __init__(self, n_entities, n_relations, n_dims_sm, rel_emb_l2):
        super(AttentionFlow, self).__init__(name='att_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims_sm

        self.relation_embedding = keras.layers.Embedding(n_relations, self.n_dims,
                                                         embeddings_regularizer=keras.regularizers.l2(rel_emb_l2),
                                                         name='relations')
        # f(hidden_con_vi, hidden_uncon_vi, rel_emb, hidden_con_vj, hidden_uncon_vj)
        self.f_transition = F([[0, 3], [0, 2, 3], [0, 4], [0, 2, 4], [1, 3], [1, 2, 3], [1, 4], [1, 2, 4]], self.n_dims,
                              activation=tf.nn.relu, output_weight=True, output_bias=True, name='f_trans')
        self.proj_con = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj_con')
        self.proj_uncon = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj_uncon')

        self.nodes_to_edges = Node2Edge()
        self.nodes_to_edges_v2 = Node2Edge_v2()

        self.neighbor_softmax = NeighborSoftmax()

        self.aggregate = Aggregate()

    def call(self, inputs, scanned_edges=None, edges_y=None, hidden_uncon=None, hidden_con=None,
             new_idx_for_memorized=None, n_memorized_and_scanned_nodes=None, training=None, tc=None):
        """ inputs (node_attention): batch_size x n_nodes
            scanned_edges (aug_scanned_edges): n_aug_scanned_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
              (1) including selfloop edges and backtrace edges
              (2) batch_size >= 1
            edges_y: n_aug_scanned_edges ( sorted according to scanned_edges )
            hidden_uncon: 1 x n_nodes x n_dims_lg
            hidden_con: n_memorized_nodes x n_dims
        """
        assert scanned_edges is not None
        assert edges_y is not None
        assert hidden_con is not None
        assert hidden_uncon is not None
        assert n_memorized_and_scanned_nodes is not None
        if tc is not None:
            t0 = time.time()

        hidden_con = self.proj_con(hidden_con)  # n_memorized_nodes x n_dims_sm

        if new_idx_for_memorized is not None:
            hidden_con = tf.scatter_nd(new_idx_for_memorized, hidden_con,
                                       tf.TensorShape((n_memorized_and_scanned_nodes, self.n_dims)))  # n_memorized_and_scanned_nodes x n_dims_sm

        hidden_uncon = self.proj_uncon(hidden_uncon)  # 1 x n_nodes x n_dims_sm

        # compute transition
        hidden_con_vi, hidden_con_vj = self.nodes_to_edges_v2(hidden_con, scanned_edges)  # n_aug_scanned_edges x n_dims_sm

        hidden_uncon_vi, hidden_uncon_vj = self.nodes_to_edges(hidden_uncon, scanned_edges)  # n_aug_scanned_edges x n_dims_sm

        rel_idx = scanned_edges[:, 3]  # n_aug_scanned_edges
        rel_emb = self.relation_embedding(rel_idx)  # n_aug_scanned_edges x n_dims_sm

        transition_logits = self.f_transition((hidden_con_vi, hidden_uncon_vi, rel_emb, hidden_con_vj, hidden_uncon_vj))  # n_aug_scanned_edges x n_dims_sm
        transition_logits = tf.reduce_sum(transition_logits, axis=1)  # n_aug_scanned_edges

        transition = self.neighbor_softmax(transition_logits, selected_edges=scanned_edges)  # n_aug_scanned_edges

        # compute transition attention
        node_attention = inputs  # batch_size x n_nodes
        idx_and_vi = tf.stack([scanned_edges[:, 0], scanned_edges[:, 1]], axis=1)  # n_aug_scanned_edges x 2
        node_attention = tf.gather_nd(node_attention, idx_and_vi)  # n_aug_scanned_edges
        trans_attention = node_attention * transition * edges_y  # n_aug_scanned_edges

        # compute new node attention
        batch_size = tf.shape(inputs)[0]
        new_node_attention = self.aggregate(trans_attention, selected_edges=scanned_edges,  # batch_size x n_nodes
                                            output_shape=(batch_size, self.n_nodes),
                                            at='vj', aggr_op='sum')
        new_node_attention_sum = tf.reduce_sum(new_node_attention, axis=1, keepdims=True)  # batch_size x 1
        new_node_attention = new_node_attention / new_node_attention_sum  # batch_size x n_nodes

        if tc is not None:
            tc['a.call'] += time.time() - t0
        # trans_attention: n_aug_scanned_edges
        # new_node_attention: batch_size x n_nodes
        return trans_attention, new_node_attention

    def get_init_node_attention(self, heads, tc=None):
        if tc is not None:
            t0 = time.time()

        with tf.name_scope(self.name):
            node_attention = tf.one_hot(heads, self.n_nodes)  # batch_size x n_nodes

        if tc is not None:
            tc['a.init'] += time.time() - t0
        return node_attention


class Model(object):
    def __init__(self, graph, hparams):
        self.graph = graph
        self.hparams = hparams

        self.sampler = Sampler(graph)
        self.uncon_flow = UnconsciousnessFlow(graph.n_entities, graph.n_relations, hparams.n_dims_lg,
                                              hparams.ent_emb_l2, hparams.rel_emb_l2)
        self.con_flow = ConsciousnessFlow(graph.n_entities, graph.n_aug_relations, hparams.n_dims,
                                          hparams.ent_emb_l2, hparams.rel_emb_l2)
        self.att_flow = AttentionFlow(graph.n_entities, graph.n_aug_relations, hparams.n_dims_sm,
                                      hparams.rel_emb_l2)
        self.hidden_uncon_past = None
        self.node_attention_trace = None

    def init_per_graph(self):
        """ initialize unconsciousness flow at the beginning of each graph (no backprop)
        """
        hidden_uncon = self.uncon_flow.get_init_hidden()  # 1 x n_nodes x n_dims_lg

        # run unconsciousness flow initially for multiple steps
        if self.hparams.init_uncon_steps_per_graph is not None:
            for _ in range(self.hparams.init_uncon_steps_per_graph):
                # candidate_edges: (pd.DataFrame) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
                candidate_edges = self.graph.get_candidate_edges()

                # edges_logits: (tf.Variable) n_full_edges
                edges_logits = self.sampler.edges_logits

                # loglog_u: (np.array) n_candidate_edges
                # sampled_edges: (pd.DataFrame) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
                loglog_u, sampled_edges = self.graph.sample_edges(candidate_edges, edges_logits,
                                                                  mode='by_eg',
                                                                  max_edges_per_eg=self.hparams.max_edges_per_example)

                # edges_y: (tf.Tensor) n_sample_edges
                edges_y = self.sampler(None, candidate_edges=candidate_edges, loglog_u=loglog_u,
                                       sampled_edges=sampled_edges, mode='by_eg')

                # selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj] sorted by (eg_idx, vi, vj)
                selected_edges = self.graph.get_selected_edges(sampled_edges)

                hidden_uncon = self.uncon_flow(hidden_uncon, selected_edges=selected_edges, edges_y=edges_y)  # 1 x n_nodes x n_dims_lg
        self.hidden_uncon_past = hidden_uncon

    def init_per_batch(self, heads, rels, tc=None):
        """ heads: batch_size
            rels: batch_size
        """
        ''' initialize unconsciousness flow at the beginning of each batch (with backprop) '''
        hidden_uncon = self.uncon_flow.get_init_hidden(hidden_past=self.hidden_uncon_past)  # 1 x n_nodes x n_dims_lg

        ''' run unconsciousness flow initially for multiple steps '''
        if self.hparams.init_uncon_steps_per_batch is not None:
            for _ in range(self.hparams.init_uncon_steps_per_batch):
                # candidate_edges: (pd.DataFrame) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
                candidate_edges = self.graph.get_candidate_edges(tc=get(tc, 'graph'))

                # loglog_u: (np.array) n_candidate_edges
                # sampled_edges: (pd.DataFrame) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
                loglog_u, sampled_edges = self.graph.sample_edges(candidate_edges, self.sampler.edges_logits,
                                                                  mode='by_eg',
                                                                  max_edges_per_eg=self.hparams.max_edges_per_example,
                                                                  tc=get(tc, 'graph'))

                # edges_y: (tf.Tensor) n_sample_edges
                edges_y = self.sampler(None, candidate_edges=candidate_edges, loglog_u=loglog_u,
                                       sampled_edges=sampled_edges, mode='by_eg', tc=get(tc, 'model'))

                # selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj] sorted by (eg_idx, vi, vj)
                selected_edges = self.graph.get_selected_edges(sampled_edges, tc=get(tc, 'graph'))

                hidden_uncon = self.uncon_flow(hidden_uncon, selected_edges=selected_edges, edges_y=edges_y,
                                               tc=get(tc, 'model'))  # 1 x n_nodes x n_dims_lg

        ''' initialize attention flow '''
        node_attention = self.att_flow.get_init_node_attention(heads)  # batch_size x n_nodes

        ''' initialize consciousness flow '''
        query_context = self.con_flow.get_query_context(heads, rels)  # batch_size x n_dims
        memorized_v = self.graph.set_init_memorized_nodes(heads)  # n_memorized_nodes (=batch_size) x 2, (eg_idx, v)
        hidden_con = self.con_flow.get_init_hidden(query_context, hidden_uncon, memorized_v)  # n_memorized_nodes x n_dims

        self.node_attention_trace = [node_attention]

        self.graph.set_past_transitions()
        self.graph.set_node_attention_li(node_attention)
        self.graph.set_attended_nodes_li()

        # hidden_uncon: 1 x n_nodes x n_dims_lg
        # hidden_con: n_memorized_nodes
        # node_attention: batch_size x n_nodes
        return hidden_uncon, hidden_con, node_attention

    def flow(self, hidden_uncon, hidden_con, node_attention, step, tc=None):
        """ hidden_uncon: 1 x n_nodes x n_dims_lg
            hidden_con: n_memorized_nodes x n_dims
            node_attention: batch_size x n_nodes
        """
        ''' run unconsciousness flow '''
        if self.hparams.simultaneous_uncon_flow:
            # candidate_edges: (pd.DataFrame) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
            candidate_edges = self.graph.get_candidate_edges(get(tc, 'graph'))

            # loglog_u: (np.array) n_candidate_edges
            # sampled_edges: (pd.DataFrame) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
            loglog_u, sampled_edges = self.graph.sample_edges(candidate_edges, self.sampler.edges_logits,
                                                              mode='by_eg',
                                                              max_edges_per_eg=self.hparams.max_edges_per_example,
                                                              tc=get(tc, 'graph'))

            # edges_y: (tf.Tensor) n_sample_edges
            edges_y = self.sampler(None, candidate_edges=candidate_edges, loglog_u=loglog_u,
                                   sampled_edges=sampled_edges, mode='by_eg', tc=get(tc, 'model'))

            # selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj] sorted by (eg_idx, vi, vj)
            selected_edges = self.graph.get_selected_edges(sampled_edges, tc=get(tc, 'graph'))

            new_hidden_uncon = self.uncon_flow(hidden_uncon, selected_edges=selected_edges, edges_y=edges_y,
                                               tc=get(tc, 'model'))  # 1 x n_nodes x n_dims_lg
        else:
            new_hidden_uncon = hidden_uncon  # 1 x n_nodes x n_dims_lg

        ''' get scanned edges '''
        # attended_nodes: (np.array) n_attended_nodes x 2, (eg_idx, vi) not sorted
        attended_nodes = self.graph.get_topk_nodes(node_attention, self.hparams.max_attended_nodes,
                                                   tc=get(tc, 'graph'))  # n_attended_nodes x 2

        # candidate_edges: (pd.DataFrame) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
        candidate_edges = self.graph.get_candidate_edges(attended_nodes=attended_nodes,
                                                         tc=get(tc, 'graph'))  # n_candidate_edges x 2

        # loglog_u: (np.array) n_candidate_edges
        # sampled_edges: (pd.DataFrame) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
        loglog_u, sampled_edges = self.graph.sample_edges(candidate_edges, self.sampler.edges_logits,
                                                          mode='by_vi',
                                                          max_edges_per_vi=self.hparams.max_edges_per_node,
                                                          tc=get(tc, 'graph'))

        # edges_y: (tf.Tensor) n_sample_edges
        edges_y = self.sampler(None, candidate_edges=candidate_edges, loglog_u=loglog_u,
                               sampled_edges=sampled_edges, mode='by_vi', tc=get(tc, 'model'))

        # scanned_edges: (np.array) n_scanned_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        scanned_edges = self.graph.get_selected_edges(sampled_edges, tc=get(tc, 'graph'))

        ''' add selfloop and backtrace edges '''
        attended_node_att = tf.gather_nd(node_attention, attended_nodes)  # n_attended_nodes
        selfloop_edges, backtrace_edges = self.graph.get_selfloop_and_backtrace(attended_nodes,
                                                                                attended_node_att,
                                                                                self.hparams.max_backtrace_edges,
                                                                                step,
                                                                                backtrace_decay=self.hparams.backtrace_decay,
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

        trans_attention, new_node_attention = self.att_flow(node_attention, scanned_edges=aug_scanned_edges,
                                                            edges_y=edges_y, hidden_uncon=hidden_uncon,
                                                            hidden_con=hidden_con, new_idx_for_memorized=new_idx_for_memorized,
                                                            n_memorized_and_scanned_nodes=n_memorized_and_scanned_nodes,
                                                            tc=get(tc, 'model'))  # n_aug_scanned_edges, batch_size x n_nodes
        self.node_attention_trace.append(new_node_attention)

        ''' get seen edges '''
        # seen_nodes: (np.array) n_seen_nodes x 2, (eg_idx, vj) unique but not sorted
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
            self.graph.add_nodes_to_memorized(seen_edges, inplace=True, tc=get(tc, 'graph'))

        if new_idx_for_memorized is not None:
            hidden_con = tf.scatter_nd(new_idx_for_memorized, hidden_con,
                                       tf.TensorShape((n_memorized_and_seen_nodes, self.hparams.n_dims)))  # n_memorized_nodes (new) x n_dims

        # seen_edges: n_seen_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        seen_edges = self.graph.set_index_over_nodes(seen_edges, memorized_and_seen, tc=get(tc, 'graph'))

        new_hidden_con = self.con_flow(hidden_con, seen_edges=seen_edges, edges_y=edges_y,
                                       trans_attention=trans_attention, node_attention=new_node_attention,
                                       hidden_uncon=hidden_uncon, memorized_nodes=self.graph.memorized_nodes,
                                       tc=get(tc, 'model'))  # n_memorized_nodes (new) x n_dims

        ''' do storing work at the end of each step '''
        self.graph.add_transitions(seen_edges, trans_attention, step)
        self.graph.add_node_attention(new_node_attention)
        self.graph.add_attended_nodes(attended_nodes)

        # new_hidden_uncon: 1 x n_nodes x n_dims_lg,
        # new_hidden_con: n_memorized_nodes x n_dims,
        # new_node_attention: batch_size x n_nodes
        return new_hidden_uncon, new_hidden_con, new_node_attention

    @property
    def regularization_loss(self):
        return tf.add_n(self.uncon_flow.losses) + \
               tf.add_n(self.con_flow.losses) + \
               tf.add_n(self.att_flow.losses)

    @property
    def trainable_variables(self):
        return self.sampler.trainable_variables + \
               self.uncon_flow.trainable_variables + \
               self.con_flow.trainable_variables + \
               self.att_flow.trainable_variables
