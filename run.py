import argparse
import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras

from model import Model
from data_env import DataEnv
import datasets

parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('--n_dims', type=int, default=64)
parser.add_argument('--l2_factor', type=float, default=0.01)
parser.add_argument('--max_sampled_edges', type=int, default=1000)
parser.add_argument('--max_attended_nodes', type=int, default=100)
parser.add_argument('--max_attended_edges', type=int, default=100)
parser.add_argument('--max_backtrace_edges', type=int, default=100)
parser.add_argument('--backtrace_decay', type=float, default=0.9)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--n_virtual_nodes', type=int, default=100)
parser.add_argument('--init_uncon_steps', type=int, default=5)
parser.add_argument('--max_steps', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.001)
default_hparams = parser.parse_args()


class Trainer(object):
    def __init__(self, model, hparams):
        self.model = model
        self.hparams = hparams
        self.optimizer = keras.optimizers.Adam(learning_rate=hparams.learning_rate)

        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_pred_loss = keras.metrics.Mean(name='train_pred_loss')
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def train_step(self, heads, rels, tails):
        with tf.GradientTape() as tape:
            hidden_uncon, hidden_con, node_attention = \
                self.model.initialize(heads, rels, init_uncon_steps=self.hparams.init_uncon_steps)
            for step in range(1, self.hparams.max_steps + 1):
                hidden_uncon, hidden_con, node_attention = \
                    self.model.flow(hidden_uncon, hidden_con, node_attention, step, stop_uncon_steps=False)

            predictions = node_attention
            pred_loss = self.loss_fn(predictions, tails)
            reg_loss = self.model.regularization_loss
            loss = pred_loss + reg_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_pred_loss(pred_loss)
        self.train_accuracy(tails, predictions)

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
            self.model.initialize(heads, rels, init_uncon_steps=self.hparams.init_uncon_steps)
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

            rank = np.where(sorted_idx == head)[0].item() + 1

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

            batch_i = 1
            for train_batch, batch_size in train_batcher(hparams.batch_size):
                heads, rels, tails = train_batch[:, 0], train_batch[:, 1], train_batch[:, 2]
                trainer.train_step(heads, rels, tails)

                train_loss, train_pred_loss, train_accuracy = trainer.metric_result()
                print('epoch: {:d} | graph: {:d} | batch: {:d} | '
                      'train_loss: {:.6f} | pred_loss: {:.6f} | accuracy: {:.6f}'.format(epoch, graph_i, batch_i,
                                                                                         train_loss.numpy(),
                                                                                         train_pred_loss.numpy(),
                                                                                         train_accuracy.numpy()))
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
    run(datasets.Countries(), copy.deepcopy(default_hparams))
