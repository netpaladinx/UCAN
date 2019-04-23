import os
from collections import defaultdict

import numpy as np


class Dataset(object):
    def __init__(self, train_path, valid_path, test_path, split_ratio=None, include_reverse=False):
        split_ratio = list(map(lambda x: float(x), split_ratio.split(':'))) if split_ratio is not None else None

        train = self._load_file(train_path)
        valid = self._load_file(valid_path)
        test = self._load_file(test_path)

        self.entity2id, self.id2entity, self.relation2id, self.id2relation = self._make_dict(train + valid + test)
        self.n_entities = len(self.entity2id)
        self.n_relations = len(self.relation2id)

        if split_ratio is not None:
            train, valid, test = self._split_into_train_valid_test(train + valid + test, split_ratio, shuffle=True)

        if include_reverse:
            train = self._add_reverse_triples(train)
            valid = self._add_reverse_triples(valid)
            test = self._add_reverse_triples(test)

            self.entity2id, self.id2entity, self.relation2id, self.id2relation = self._make_dict(train + valid + test)
            self.n_entities = len(self.entity2id)
            self.n_relations = len(self.relation2id)

        self.train = self._convert_to_id(train)
        self.valid = self._convert_to_id(valid)
        self.test = self._convert_to_id(test)

    def _load_file(self, filepath):
        """ Return: string triples (head, tail, relation)
        """
        triples = []
        with open(filepath) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                triples.append((h, t, r))
        return triples

    def _make_dict(self, triples):
        ent2id, rel2id = {}, {}
        id2ent, id2rel = {}, {}
        for h, t, r in triples:
            ent2id.setdefault(h, len(ent2id))
            id2ent[ent2id[h]] = h
            ent2id.setdefault(t, len(ent2id))
            id2ent[ent2id[t]] = t
            rel2id.setdefault(r, len(rel2id))
            id2rel[rel2id[r]] = r
        return ent2id, id2ent, rel2id, id2rel

    def _split_into_train_valid_test(self, triples, split_ratio, shuffle=True):
        if shuffle:
            shuffled_idx = np.random.permutation(len(triples))
            triples = [triples[i] for i in shuffled_idx]
        l_total = len(triples)
        l_train = int(l_total * split_ratio[0] / (split_ratio[0] + split_ratio[1] + split_ratio[2]))
        l_valid = int((l_total - l_train) * split_ratio[1] / (split_ratio[1] + split_ratio[2]))
        train = triples[:l_train]
        valid = triples[l_train:l_train+l_valid]
        test = triples[l_train+l_valid:]
        return train, valid, test

    def _add_reverse_triples(self, triples):
        return triples + [(t, h, '_' + r) for h, t, r in triples]

    def _convert_to_id(self, triples):
        return np.array([(self.entity2id[h], self.entity2id[t], self.relation2id[r])
                         for h, t, r in triples], dtype='int32')

    def stats(self):
        ent_train = set()
        rel_train = set()
        n_rec = 0
        with open(self.train_path) as fin:
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
        with open(self.valid_path) as fin:
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
        with open(self.test_path) as fin:
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
        with open(self.train_path) as fin:
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


class FB237(Dataset):
    path = 'data/kbc/FB237'

    def __init__(self):
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(FB237, self).__init__(train_path, valid_path, test_path, include_reverse=True)


class YAGO310(Dataset):
    path = 'data/kbc/YAGO3-10'

    def __init__(self):
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(YAGO310, self).__init__(train_path, valid_path, test_path, )


class Countries(Dataset):
    path = 'data/MINERVA/countries'

    def __init__(self, subname='_S1'):
        self.path = Countries.path + subname
        train_path = os.path.join(self.path, 'train.txt')
        valid_path = os.path.join(self.path, 'dev.txt')
        test_path = os.path.join(self.path, 'test.txt')
        super(Countries, self).__init__(train_path, valid_path, test_path, )


if __name__ == '__main__':
    yago310 = YAGO310()
    yago310.stats()
