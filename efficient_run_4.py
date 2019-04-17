import argparse
import copy
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras

from efficient_model_3 import Model
from efficient_data_env_3 import DataEnv
import datasets

parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batch_size', type=int, default=20)
parser.add_argument('--n_dims', type=int, default=50)
parser.add_argument('--ent_emb_l2', type=float, default=0.)
parser.add_argument('--rel_emb_l2', type=float, default=0.)
parser.add_argument('--max_sampled_edges', type=int, default=10000)
parser.add_argument('--max_attended_nodes', type=int, default=200)
parser.add_argument('--max_attended_edges', type=int, default=2000)
parser.add_argument('--max_backtrace_edges', type=int, default=200)
parser.add_argument('--backtrace_decay', type=float, default=0.9)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--n_virtual_nodes', type=int, default=10)
parser.add_argument('--init_uncon_steps_per_graph', type=int, default=10)
parser.add_argument('--init_uncon_steps_per_batch', type=int, default=2)
parser.add_argument('--simultaneous_uncon_flow', action='store_true', default=False)
parser.add_argument('--max_steps', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--dataset', default='FB237')
parser.add_argument('--timer', action='store_false', default=True)
default_hparams = parser.parse_args()


class Trainer(object):
    def __init__(self, model, hparams):
        self.model = model
        self.hparams = hparams
        self.optimizer = keras.optimizers.Adam(learning_rate=hparams.learning_rate)

        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_pred_loss = keras.metrics.Mean(name='train_pred_loss')
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def train_step(self, heads, rels, tails, time_cost=None):
        with tf.GradientTape() as tape:
            hidden_uncon, hidden_con, node_attention = \
                self.model.init_per_batch(heads, rels, time_cost=time_cost)
            for step in range(1, self.hparams.max_steps + 1):
                hidden_uncon, hidden_con, node_attention = \
                    self.model.flow(hidden_uncon, hidden_con, node_attention, step, time_cost=time_cost)
            self.model.past_hidden_uncon = hidden_uncon

            predictions = node_attention
            pred_loss = self.loss_fn(predictions, tails)
            reg_loss = self.model.regularization_loss
            loss = pred_loss + reg_loss

        if time_cost is not None:
            t0 = time.time()
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if time_cost is not None:
            time_cost['grad']['comp'] += time.time() - t0

        if time_cost is not None:
            t0 = time.time()
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if time_cost is not None:
            time_cost['grad']['apply'] += time.time() - t0

        self.train_loss(loss)
        self.train_pred_loss(pred_loss)
        self.train_accuracy(tails, predictions)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, axis=1), tails), tf.float32))
        return loss, pred_loss, accuracy

    def loss_fn(self, predictions, tails):
        pred_idx = tf.stack([tf.range(0, tf.shape(tails)[0]), tails], axis=1)
        pred_prob = tf.gather_nd(predictions, pred_idx)
        pred_loss = tf.reduce_mean(- tf.math.log(pred_prob + 1e-20))
        return pred_loss

    def reset_metric(self):
        self.train_accuracy.reset_states()

    def metric_result(self):
        return self.train_loss.result(), self.train_pred_loss.result(), self.train_accuracy.result()


class Evaluator(object):
    def __init__(self, model, data_env, hparams):
        self.model = model
        self.data_env = data_env
        self.hparams = hparams

        self.heads = []
        self.relations = []
        self.predictions = []
        self.targets = []

    def eval_step(self, heads, rels, tails):
        hidden_uncon, hidden_con, node_attention = \
            self.model.init_per_batch(heads, rels, init_uncon_steps=self.hparams.init_uncon_steps)
        for step in range(1, self.hparams.max_steps + 1):
            hidden_uncon, hidden_con, node_attention = \
                self.model.flow(hidden_uncon, hidden_con, node_attention, step, stop_uncon_steps=False)

        self.heads.append(heads)
        self.relations.append(rels)
        self.predictions.append(node_attention.numpy())
        self.targets.append(tails)

    def reset_metric(self):
        self.heads = []
        self.relations = []
        self.predictions = []
        self.targets = []

    def metric_result(self):
        heads = np.concatenate(self.heads, axis=0)
        relations = np.concatenate(self.relations, axis=0)
        predictions = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        return self._calc_metrics(heads, relations, predictions, targets, self.data_env.filter_pool)

    def _calc_metrics(self, heads, relations, predictions, targets, filter_pool):
        hit_1, hit_3, hit_5, hit_10, mr, mrr = 0., 0., 0., 0., 0., 0.

        n_preds = predictions.shape[0]
        for i in range(n_preds):
            head = heads[i]
            rel = relations[i]
            tar = targets[i]
            pred = predictions[i]
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

        hit_1 /= n_preds
        hit_3 /= n_preds
        hit_5 /= n_preds
        hit_10 /= n_preds
        mr /= n_preds
        mrr /= n_preds

        return hit_1, hit_3, hit_5, hit_10, mr, mrr


def reset_time_cost(hparams):
    if hparams.timer:
        return {'model': defaultdict(float), 'graph': defaultdict(float), 'grad': defaultdict(float)}
    else:
        return None


def str_time_cost(time_cost):
    if time_cost is not None:
        model_tc = ', '.join('m.{} {:3f}'.format(k, v) for k, v in time_cost['model'].items())
        graph_tc = ', '.join('g.{} {:3f}'.format(k, v) for k, v in time_cost['graph'].items())
        grad_tc = ', '.join('d.{} {:3f}'.format(k, v) for k, v in time_cost['grad'].items())
        return model_tc + ', ' + graph_tc + ', ' + grad_tc
    else:
        return ''


def run(dataset, hparams):
    data_env = DataEnv(dataset, hparams)
    model = Model(data_env.graph, hparams)
    trainer = Trainer(model, hparams)
    evaluator = Evaluator(model, data_env, hparams)

    for epoch in range(1, hparams.max_epochs + 1):
        trainer.reset_metric()
        evaluator.reset_metric()

        n_train = data_env.n_test
        graph_i = 1
        for train_batcher, graph in data_env.draw_train(n_train):

            model.init_per_graph()

            batch_i = 1
            for train_batch, batch_size in train_batcher(hparams.batch_size):
                t0 = time.time()
                time_cost = reset_time_cost(hparams)

                heads, rels, tails = train_batch[:, 0], train_batch[:, 1], train_batch[:, 2]
                cur_train_loss, cur_pred_loss, cur_accuracy = trainer.train_step(heads, rels, tails,
                                                                                 time_cost=time_cost)

                train_loss, pred_loss, accuracy = trainer.metric_result()
                dt = time.time() - t0

                print('{:d}, {:d}, {:d} | tr_ls: {:.4f} ({:.4f}) | pr_ls: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) |'
                      ' t: {:3f} | {}'.format(epoch, graph_i, batch_i,
                                              train_loss.numpy(), cur_train_loss,
                                              pred_loss.numpy(), cur_pred_loss,
                                              accuracy.numpy(), cur_accuracy,
                                              dt,
                                              str_time_cost(time_cost)))
                batch_i += 1

            graph_i += 1

        data_env.graph.use_full_edges()

        valid_batcher = data_env.get_valid_batcher()
        for valid_batch, batch_size in valid_batcher(hparams.batch_size):
            heads, rels, tails = valid_batch[:, 0], valid_batch[:, 1], valid_batch[:, 2]
            evaluator.eval_step(heads, rels, tails)

        hit_1, hit_3, hit_5, hit_10, mr, mrr = evaluator.metric_result()
        print('epoch: {:d} | hit_1: {:.6f} | hit_3: {:.6f} | hit_5: {:.6f} | hit_10: {:.6f} | mr: {:.1f} | mmr: {:6f}'
              .format(epoch, hit_1, hit_3, hit_5, hit_10, mr, mrr))

        # test_batcher = data_env.get_test_batcher()


if __name__ == '__main__':
    hparams = copy.deepcopy(default_hparams)
    print(hparams)

    dataset = getattr(datasets, hparams.dataset)()
    run(dataset, hparams)
