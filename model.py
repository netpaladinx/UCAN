import numpy as np

import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras


class F(keras.layers.Layer):
    def __init__(self, interact, n_dims, use_bias=True, activation=None, output_weight=False, output_bias=False):
        super(F, self).__init__()
        self.interact = interact
        self.use_bias = use_bias
        self.activation = activation
        self.output_weight = output_weight
        self.output_bias = output_bias

        n_ws = len(self.interact)
        self.ws = self.add_weight(shape=(n_ws, n_dims), initializer=keras.initializers.VarianceScaling())
        if self.use_bias:
            self.b = self.add_weight(shape=(n_dims,), initializer=keras.initializers.zeros())

        if self.output_weight:
            self.out_w = self.add_weight(shape=(n_dims,), initializer=keras.initializers.VarianceScaling())
        if self.output_bias:
            self.out_b = self.add_weight(shape=(n_dims,), initializer=keras.initializers.zeros())

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
    def __init__(self, n_dims, use_bias=True, activation=None, output_weight=False, output_bias=False, residual=True):
        super(G, self).__init__()
        self.output_weight = output_weight
        self.residual = residual
        self.dense = keras.layers.Dense(n_dims, activation=activation, use_bias=use_bias)
        if self.output_weight:
            self.dense_out = keras.layers.Dense(n_dims, use_bias=output_bias)

    def call(self, inputs, training=None):
        outputs = self.dense(inputs)
        if self.output_weight:
            outputs = self.dense_out(outputs)
        if self.residual:
            outputs = inputs + outputs
        return outputs


class N2E(keras.layers.Layer):
    def call(self, inputs, selected_edges=None, training=None):
        """ inputs (hidden): batch_size x n_nodes x n_dims
            selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
        """
        assert selected_edges is not None
        hidden = inputs
        idx_and_vi = tf.stack([selected_edges[:, 0], selected_edges[:, 1]], axis=1)  # n_selected_edges x 2
        idx_and_vj = tf.stack([selected_edges[:, 0], selected_edges[:, 2]], axis=1)  # n_selected_edges x 2
        hidden_vi = tf.gather_nd(hidden, idx_and_vi)  # n_selected_edges x n_dims
        hidden_vj = tf.gather_nd(hidden, idx_and_vj)  # n_selected_edges x n_dims
        return hidden_vi, hidden_vj


class Aggregate(keras.layers.Layer):
    def call(self, inputs, selected_edges=None, output_shape=None, at='vj', aggr_op='mean', training=None):
        """ inputs (edge_vec): n_seleted_edges x ...
            selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
            output_shape: (batch_size, n_nodes, ...)
        """
        assert selected_edges is not None
        assert output_shape is not None
        edge_vec = inputs
        if at == 'vi':
            idx_vi = selected_edges[:, -2]  # n_selected_edges
            aggr_op = tf.math.segment_mean if aggr_op == 'mean' else \
                tf.math.segment_sum if aggr_op == 'sum' else \
                tf.math.segment_max if aggr_op == 'max' else None
            edge_vec_aggr = aggr_op(edge_vec, idx_vi)  # (max_idx_vi+1) x ...
            idx_and_vi = tf.stack([selected_edges[:, 0], selected_edges[:, 1]], axis=1)  # n_selected_edges x 2
            idx_and_vi = tf.cast(tf.math.segment_max(idx_and_vi, idx_vi), tf.int32)  # (max_id_vi+1) x 2
            edge_vec_aggr = tf.scatter_nd(idx_and_vi, edge_vec_aggr, output_shape)  # batch_size x n_nodes x ...
        elif at == 'vj':
            idx_vj = selected_edges[:, -1]  # n_selected_edges
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


class Normalize(keras.layers.Layer):
    def call(self, inputs, selected_edges=None, at='vi', training=None):
        """ inputs (edge_vec): n_seleted_edges x ...
            selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
        """
        assert selected_edges is not None
        edge_vec = inputs
        if at == 'vi':
            idx_vi = selected_edges[:, -2]  # n_selected_edges
            edge_vec_max = tf.math.segment_max(edge_vec, idx_vi)  # (max_idx_vi+1) x ...
            edge_vec_max = tf.gather(edge_vec_max, idx_vi)  # n_selected_edges x ...
            edge_vec_exp = tf.math.exp(edge_vec - edge_vec_max)  # n_selected_edges x ...
            edge_vec_expsum = tf.math.segment_sum(edge_vec_exp, idx_vi)  # (max_idx_vi+1) x ...
            edge_vec_expsum = tf.gather(edge_vec_expsum, idx_vi)  # n_selected_edges x ...
            edge_vec_norm = edge_vec_exp / edge_vec_expsum  # n_selected_edges x ...
        elif at == 'vj':
            idx_vj = selected_edges[:, -1]  # n_selected_edges
            max_idx_vj = tf.reduce_max(idx_vj)
            edge_vec_max = tf.math.unsorted_segment_max(edge_vec, idx_vj, max_idx_vj + 1)  # (max_idx_vi+1) x ...
            edge_vec_max = tf.gather(edge_vec_max, idx_vj)  # n_selected_edges x ...
            edge_vec_exp = tf.math.exp(edge_vec - edge_vec_max)  # n_selected_edges x ...
            edge_vec_expsum = tf.math.unsorted_segment_sum(edge_vec_exp, idx_vj, max_idx_vj + 1)  # (max_idx_vi+1) x ...
            edge_vec_expsum = tf.gather(edge_vec_expsum, idx_vj)  # n_selected_edges x ...
            edge_vec_norm = edge_vec_exp / edge_vec_expsum  # n_selected_edges x ...
        else:
            raise ValueError('Invalid `at`')
        return edge_vec_norm


class Sampler(keras.Model):
    def __init__(self, graph):
        super(Sampler, self).__init__()
        self.edges_p = self.add_weight(shape=(graph.n_full_edges,),
                                       initializer=keras.initializers.constant(self._initialize(graph)))

    def _initialize(self, graph):
        p_init = np.zeros((graph.n_full_edges,), np.float32)
        for e_id, vi, rel, vj in graph.full_edges:
            p_init[e_id] = (1. / graph.count(vi)) * (1. / graph.count((vi, rel))) * (1. / graph.count((vi, rel, vj)))
        return p_init

    def call(self, inputs, max_edges=None, training=None):
        """ inputs: n_candidate_edges x 2 ( inputs[i] = (eg_idx, edge_id) )
        """
        assert max_edges is not None

        eg_idx, edge_id = tf.split(inputs, 2, axis=1)  # n_candidate_edges, n_candidate_edges
        batch_size = tf.reduce_max(eg_idx) + 1
        n_candidate_edges = tf.shape(inputs)[0]

        prob = tf.gather(self.edges_p, edge_id)  # n_candidate_edges

        def cond(t, start, outputs, edges_y):
            return tf.less(t, batch_size)

        def body(t, start, outputs, edges_y):
            mask = tf.equal(eg_idx, t)
            e_id = tf.boolean_mask(edge_id, mask)
            p = tf.boolean_mask(prob, mask)

            y, idx, n_e = self._gumbel_softmax(p, k=max_edges)
            e = tf.gather(e_id, idx)
            eg = tf.tile(tf.expand_dims(t, 0), [n_e])
            eg_e = tf.stack([eg, e], axis=1)

            idx_range = tf.range(start, start + n_e)
            outputs = outputs.scatter(idx_range, eg_e)
            edges_y = edges_y.scatter(idx_range, y)

            return t + 1, start + n_e, outputs, edges_y

        outputs = tf.TensorArray(tf.int32, size=n_candidate_edges, element_shape=tf.TensorShape((2,)))
        edges_y = tf.TensorArray(tf.float32, size=n_candidate_edges, element_shape=tf.TensorShape(()))
        t, n_sampled_edges, outputs, edges_y = tf.while_loop(cond, body,
                                                             (tf.constant(0), tf.constant(0), outputs, edges_y))

        idx_range = tf.range(0, n_sampled_edges)
        outputs = outputs.gather(idx_range)
        edges_y = edges_y.gather(idx_range)

        # outputs: n_sampled_edges x 2 ( int32, outputs[i] = (eg_idx, edge_id) )
        # edges_y: n_sampled_edges ( float32, edges_y[i] = y )
        return outputs, edges_y

    def _gumbel_softmax(self, logits, k=1, temperature=1., hard=True):
        """ logits: n
        """
        eps = 1e-20
        y = logits - tf.math.log(- tf.math.log(tf.random.uniform(tf.shape(logits)) + eps) + eps)
        y = tf.math.softmax(y / temperature)
        k = tf.math.minimum(k, tf.shape(y)[-1])
        y, ind = tf.math.top_k(y, k=k)
        if hard:
            y_hard = tf.ones_like(y)
            y = tf.stop_gradient(y_hard - y) + y

        # y: k (k <= n)
        # ind: k (k <= n)
        return y, ind, k


class UnconsciousnessFlow(keras.Model):
    def __init__(self, n_entities, n_relations, n_dims, l2_factor):
        """ n_entities: including virtual nodes
            n_relations: including 'virtual' but not 'selfloop' and 'backtrace'
        """
        super(UnconsciousnessFlow, self).__init__()
        self.n_nodes = n_entities
        self.n_dims = n_dims

        self.entity_embedding = keras.layers.Embedding(n_entities, self.n_dims,
                                                       embeddings_regularizer=keras.regularizers.l2(l2_factor))
        self.relation_embedding = keras.layers.Embedding(n_relations, self.n_dims,
                                                         embeddings_regularizer=keras.regularizers.l2(l2_factor))

        # f(message_aggr, hidden, ent_emb)
        self.f_hidden = F([[0], [0, 1], [0, 2], [1], [2], [1, 2]], self.n_dims,
                          activation=tf.nn.relu, output_weight=True, output_bias=True)
        self.g_hidden = G(self.n_dims, activation=tf.nn.relu, output_weight=True, output_bias=True)

        self.nodes_to_edges = N2E()

        # f(hidden_vi, rel_emb, hidden_vj)
        self.f_message = F([[0], [0, 1], [0, 1, 2]], self.n_dims,
                           activation=tf.nn.relu, output_weight=True, output_bias=True)
        self.g_message = G(self.n_dims, activation=tf.nn.relu, output_weight=True, output_bias=True)

        self.aggregate = Aggregate()

        ent_idx = tf.expand_dims(tf.range(0, self.n_nodes), axis=0)  # 1 x n_nodes
        self.ent_emb = self.entity_embedding(ent_idx)  # 1 x n_nodes x n_dims

    def call(self, inputs, selected_edges=None, edges_y=None, training=None):
        """ inputs (hidden): 1 x n_nodes x n_dims
            selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
                * batch_size = 1
            edges_y: n_selected_edges ( sorted according to selected_edges )
        """
        assert selected_edges is not None
        assert edges_y is not None

        # compute unconscious messages
        hidden = inputs
        hidden_vi, hidden_vj = self.nodes_to_edges(hidden, selected_edges=selected_edges)  # n_selected_edges x n_dims
        rel_idx = selected_edges[:, 3]  # n_selected_edges
        rel_emb = self.relation_embedding(rel_idx)  # n_selected_edges x n_dims
        message = self.f_message((hidden_vi, rel_emb, hidden_vj))  # n_selected_edges x n_dims
        message = self.g_message(message)  # n_selected_edges x n_dims
        message = tf.expand_dims(edges_y, 1) * message

        # aggregate unconscious messages
        message_aggr = self.aggregate(message, selected_edges=selected_edges,
                                      output_shape=(1, self.n_nodes, self.n_dims))

        # update unconscious states
        update = self.f_hidden((message_aggr, hidden, self.ent_emb))  # 1 x n_nodes x n_dims
        update = self.g_hidden(update)
        hidden = hidden + update
        return hidden  # 1 x n_nodes x n_dims

    def get_initial_hidden(self):
        zeros = tf.zeros((1, self.n_nodes, self.n_dims))  # 1 x n_nodes x n_dims
        hidden_init = self.f_hidden((zeros, zeros, self.ent_emb))
        hidden_init = self.g_hidden(hidden_init)
        return hidden_init  # 1 x n_nodes x n_dims


class ConsciousnessFlow(keras.Model):
    def __init__(self, n_entities, n_relations, n_dims, l2_factor):
        """ n_entities: including virtual nodes
            n_relations: including 'virtual', 'selfloop' and 'backtrace'
        """
        super(ConsciousnessFlow, self).__init__()
        self.n_nodes = n_entities
        self.n_dims = n_dims

        self.entity_embedding = keras.layers.Embedding(n_entities, self.n_dims,
                                                       embeddings_regularizer=keras.regularizers.l2(l2_factor))
        self.relation_embedding = keras.layers.Embedding(n_relations, self.n_dims,
                                                         embeddings_regularizer=keras.regularizers.l2(l2_factor))

        # f(head_emb, rel_emb)
        self.f_query = F([[0], [1], [0,1]], self.n_dims,
                         activation=tf.nn.relu, output_weight=True, output_bias=True)
        self.g_query = G(self.n_dims, activation=tf.nn.relu, output_weight=True, output_bias=True)

        # f(message, hidden_uncon, hidden, ent_emb)
        self.f_hidden = F([[0], [0, 2], [0, 3], [1], [1, 2], [1, 3], [2], [3], [2, 3]], self.n_dims,
                          activation=tf.nn.relu, output_weight=True, output_bias=True)
        self.g_hidden = G(self.n_dims, activation=tf.nn.relu, output_weight=True, output_bias=True)

        ent_idx = tf.expand_dims(tf.range(0, self.n_nodes), axis=0)  # 1 x n_nodes
        self.ent_emb = self.entity_embedding(ent_idx)  # 1 x n_nodes x n_dims

        self.nodes_to_edges = N2E()

        # f(hidden_vi, rel_emb, hidden_vj)
        self.f_message = F([[0], [0, 1], [0, 1, 2]], self.n_dims,
                           activation=tf.nn.relu, output_weight=True, output_bias=True)
        self.g_message = G(self.n_dims, activation=tf.nn.relu, output_weight=True, output_bias=True)

        # f(trans_attention, message)
        self.f_attended_message = F([[0, 1]], self.n_dims, use_bias=False)

        self.aggregate = Aggregate()

    def call(self, inputs, selected_edges=None, edges_y=None, trans_attention=None, node_attention=None,
             hidden_uncon=None, training=None):
        """ inputs (hidden): batch_size x n_nodes x n_dims
            selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
                * including selfloop edges and backtrace edges
                * batch_size >= 1
            edges_y: n_selected_edges ( sorted according to selected_edges )
            trans_attention: n_selected_edges ( sorted according to selected_edges )
            node_attention: batch_size x n_nodes
            hidden_uncon: 1 x n_nodes x n_dims
        """
        assert selected_edges is not None
        assert edges_y is not None
        assert trans_attention is not None
        assert node_attention is not None
        assert hidden_uncon is not None

        # compute conscious messages
        hidden = inputs
        hidden_vi, hidden_vj = self.nodes_to_edges(hidden, selected_edges)  # n_selected_edges x n_dims
        rel_idx = selected_edges[:, 3]  # n_selected_edges
        rel_emb = self.relation_embedding(rel_idx)  # n_selected_edges x n_dims
        message = self.f_message((hidden_vi, rel_emb, hidden_vj))  # n_selected_edges x n_dims
        message = self.g_message(message)  # n_selected_edges x n_dims
        message = tf.expand_dims(edges_y, 1) * message

        # attend conscious messages
        message = self.f_attended_message((tf.expand_dims(trans_attention, 1), message))  # n_selected_edges x n_dims

        # aggregate conscious messages
        batch_size = tf.shape(inputs)[0]
        message_aggr = self.aggregate(message, selected_edges=selected_edges,
                                      output_shape=(batch_size, self.n_nodes, self.n_dims))  # batch_size x n_nodes x n_dims

        # update conscious messages
        hidden_uncon = hidden_uncon * tf.expand_dims(node_attention, 2)  # batch_size x n_nodes x n_dims
        ent_emb = tf.tile(self.ent_emb, [batch_size, 1, 1])  # batch_size x n_nodes x n_dims
        update = self.f_hidden((message_aggr, hidden_uncon, hidden, ent_emb))  # batch_size x n_nodes x n_dims
        update = self.g_hidden(update)
        hidden = hidden + update
        return hidden  # batch_size x n_nodes x n_dims

    def get_query_context(self, heads, rels):
        """ heads: batch_size
            rels: batch_size
        """
        head_emb = self.entity_embedding(heads)  # batch_size x n_dims
        rel_emb = self.relation_embedding(rels)  # batch_size x n_dims
        query_context = self.f_query((head_emb, rel_emb))
        query_context = self.g_query(query_context)
        return query_context  # batch_size x n_dims

    def get_initial_hidden(self, query_context, hidden_uncon, node_attention):
        """ query_context: batch_size x n_dims
            hidden_uncon: 1 x n_nodes x n_dims
            node_attention: batch_size x n_nodes
        """
        batch_size = tf.shape(query_context)[0]
        message = tf.expand_dims(query_context, 1) * tf.expand_dims(node_attention, 2)  # batch_size x n_nodes x n_dims
        hidden_uncon = hidden_uncon * tf.expand_dims(node_attention, 2)  # batch_size x n_nodes x n_dims
        zeros = tf.zeros_like(hidden_uncon)  # batch_size x n_nodes x n_dims
        ent_emb = tf.tile(self.ent_emb, [batch_size, 1, 1])  # batch_size x n_nodes x n_dims
        hidden_init = self.f_hidden((message, hidden_uncon, zeros, ent_emb))
        hidden_init = self.g_hidden(hidden_init)
        return hidden_init  # batch_size x n_nodes x n_dims


class AttentionFlow(keras.Model):
    def __init__(self, n_entities, n_relations, n_dims, l2_factor):
        super(AttentionFlow, self).__init__()
        self.n_nodes = n_entities
        self.n_dims = n_dims

        self.relation_embedding = keras.layers.Embedding(n_relations, self.n_dims,
                                                         embeddings_regularizer=keras.regularizers.l2(l2_factor))

        # f(hidden_con_vi, rel_emb, hidden_con_vj, hidden_uncon_vj)
        self.f_transition = F([[0, 2], [0, 1, 2], [0, 3], [0, 1, 3]], self.n_dims,
                              activation=tf.nn.relu, output_weight=True, output_bias=True)

        self.nodes_to_edges = N2E()

        self.normalize = Normalize()

        self.aggregate = Aggregate()

    def call(self, inputs, selected_edges=None, edges_y=None, hidden_con=None, hidden_uncon=None, training=None):
        """ inputs (node_attention): batch_size x n_nodes
            selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
                * including selfloop edges and backtrace edges
                * batch_size >= 1
            edges_y: n_selected_edges ( sorted according to selected_edges )
            hidden_con: batch_size x n_nodes x n_dims
            hidden_uncon: 1 x n_nodes x n_dims
        """
        assert selected_edges is not None
        assert edges_y is not None
        assert hidden_con is not None
        assert hidden_uncon is not None

        # compute transition
        hidden_con_vi, hidden_con_vj = self.nodes_to_edges(hidden_con, selected_edges)  # n_selected_edges x n_dims

        batch_size = tf.shape(inputs)[0]
        hidden_uncon = tf.tile(hidden_uncon, [batch_size, 1, 1])  # batch_size x n_nodes x n_dims
        _, hidden_uncon_vj = self.nodes_to_edges(hidden_uncon, selected_edges)  # n_selected_edges x n_dims

        rel_idx = selected_edges[:, 3]  # n_selected_edges
        rel_emb = self.relation_embedding(rel_idx)  # n_selected_edges x n_dims

        transition_logits = self.f_transition((hidden_con_vi, rel_emb, hidden_con_vj, hidden_uncon_vj))  # n_selected_edges x n_dims
        transition_logits = tf.reduce_sum(transition_logits, axis=1)  # n_selected_edges

        transition = self.normalize(transition_logits, selected_edges=selected_edges)  # n_selected_edges

        # compute transition attention
        node_attention = inputs  # batch_size x n_nodes
        idx_and_vi = tf.stack([selected_edges[:, 0], selected_edges[:, 1]], axis=1)  # n_selected_edges x 2
        node_attention = tf.gather_nd(node_attention, idx_and_vi)  # n_selected_edges
        trans_attention = node_attention * transition * edges_y  # n_selected_edges

        # compute new node attention
        batch_size = tf.shape(inputs)[0]
        new_node_attention = self.aggregate(trans_attention, selected_edges=selected_edges,  # batch_size x n_nodes
                                            output_shape=(batch_size, self.n_nodes),
                                            at='vj', aggr_op='sum')
        return trans_attention, new_node_attention

    def get_initial_node_attention(self, heads):
        node_attention = tf.one_hot(heads, self.n_nodes)  # batch_size x n_nodes
        return node_attention


class Model(object):
    def __init__(self, graph, hparams):
        self.graph = graph
        self.hparams = hparams

        self.sampler = Sampler(graph)
        self.uncon_flow = UnconsciousnessFlow(graph.n_entities, graph.n_relations, hparams.n_dims, hparams.l2_factor)
        self.con_flow = ConsciousnessFlow(graph.n_entities, graph.n_aug_relations, hparams.n_dims, hparams.l2_factor)
        self.att_flow = AttentionFlow(graph.n_entities, graph.n_aug_relations, hparams.n_dims, hparams.l2_factor)

    def initialize(self, heads, rels, init_uncon_steps=None):
        hidden_uncon = self.uncon_flow.get_initial_hidden()  # 1 x n_nodes x n_dims

        if init_uncon_steps is not None:
            for _ in range(init_uncon_steps):
                # sampled_edges: n_sampled_edges x 2 ( int32, outputs[i] = (eg_idx, edge_id) )
                # edges_y: n_sampled_edges ( float32, edges_y[i] = y )
                sampled_edges, edges_y = self.sampler(self.graph.get_graph_edges(),
                                                      max_edges=self.hparams.max_sampled_edges)

                # selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj),
                #   sorted by (idx, vi, vj), batch_size = 1 )
                selected_edges, sorted_idx = self.graph.get_selected_edges(sampled_edges)
                edges_y = tf.gather(edges_y, sorted_idx)

                hidden_uncon = self.uncon_flow(hidden_uncon, selected_edges=selected_edges, edges_y=edges_y)

        node_attention = self.att_flow.get_initial_node_attention(heads)

        query_context = self.con_flow.get_query_context(heads, rels)
        hidden_con = self.con_flow.get_initial_hidden(query_context, hidden_uncon, node_attention)

        self.graph.reset_past_transitions()

        return hidden_uncon, hidden_con, node_attention

    def flow(self, hidden_uncon, hidden_con, node_attention, step, stop_uncon_steps=False):
        """ hidden_uncon: 1 x n_nodes x n_dims
            hidden_con: batch_size x n_nodes x n_dims
            node_attention: batch_size x n_nodes
        """
        if not stop_uncon_steps:
            sampled_edges, edges_y = self.sampler(self.graph.get_graph_edges(),
                                                  max_edges=self.hparams.max_sampled_edges)
            selected_edges, sorted_idx = self.graph.get_selected_edges(sampled_edges)
            edges_y = tf.gather(edges_y, sorted_idx)
            new_hidden_uncon = self.uncon_flow(hidden_uncon, selected_edges=selected_edges, edges_y=edges_y)
        else:
            new_hidden_uncon = hidden_uncon

        attended_nodes = self.graph.get_attended_nodes(node_attention, self.hparams.max_attended_nodes)  # n_attended_nodes x 2
        attended_edges = self.graph.get_attended_edges(attended_nodes)  # n_attended_edges x 2
        sampled_edges, edges_y = self.sampler(attended_edges, max_edges=self.hparams.max_attended_edges)

        selected_edges, sorted_idx = self.graph.get_selected_edges(sampled_edges)  # selected_edges: n_selected_edges x 6
        edges_y = tf.gather(edges_y, sorted_idx)

        attended_node_attention = tf.gather_nd(node_attention, attended_nodes)  # n_attended_nodes
        selfloop_edges, backtrace_edges = self.graph.get_selfloop_and_backtrace(attended_nodes,
                                                                                attended_node_attention,
                                                                                self.hparams.max_backtrace_edges,
                                                                                step,
                                                                                backtrace_decay=self.hparams.backtrace_decay)

        aug_selected_edges, new_idx_for_edges_y, rest_idx = self.graph.get_union_edges(selected_edges,
                                                                                       selfloop_edges,
                                                                                       backtrace_edges)
        edges_y = tf.scatter_nd(new_idx_for_edges_y, edges_y, tf.TensorShape((aug_selected_edges.shape[0],)))
        rest_y = tf.scatter_nd(rest_idx, tf.ones((rest_idx.shape[0],)), tf.TensorShape((aug_selected_edges.shape[0],)))
        edges_y = edges_y + rest_y

        trans_attention, new_node_attention = self.att_flow(node_attention, selected_edges=aug_selected_edges,
                                                            edges_y=edges_y, hidden_con=hidden_con,
                                                            hidden_uncon=hidden_uncon)

        new_hidden_con = self.con_flow(hidden_con, selected_edges=aug_selected_edges, edges_y=edges_y,
                                       trans_attention=trans_attention, node_attention=node_attention,
                                       hidden_uncon=hidden_uncon)

        selected_trans_att = tf.gather_nd(trans_attention, new_idx_for_edges_y)
        self.graph.store_transitions(selected_edges, selected_trans_att, step)

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
