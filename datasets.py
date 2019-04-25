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

        self.reversed_rel_dct = None
        if include_reverse:
            train = self._add_reverse_triples(train)
            valid = self._add_reverse_triples(valid)
            test = self._add_reverse_triples(test)

            self.entity2id, self.id2entity, self.relation2id, self.id2relation = self._make_dict(train + valid + test)
            self.n_entities = len(self.entity2id)
            self.n_relations = len(self.relation2id)
            self.reversed_rel_dct = self._get_reversed_relation_dict(self.relation2id)

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

    def _get_reversed_relation_dict(self, relation2id):
        return {id: relation2id['_' + rel if rel[0] != '_' else rel[1:]] for rel, id in relation2id.items()}

    def _convert_to_id(self, triples):
        return np.array([(self.entity2id[h], self.entity2id[t], self.relation2id[r])
                         for h, t, r in triples], dtype='int32')


class FB237(Dataset):
    path = 'data/kbc/FB237'

    def __init__(self, include_reverse=True):
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(FB237, self).__init__(train_path, valid_path, test_path, include_reverse=include_reverse)


class YAGO310(Dataset):
    path = 'data/kbc/YAGO3-10'

    def __init__(self):
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(YAGO310, self).__init__(train_path, valid_path, test_path, )


class Countries(Dataset):
    path = 'data/MINERVA/countries'

    def __init__(self, subname='_S1', include_reverse=True):
        self.path = Countries.path + subname
        train_path = os.path.join(self.path, 'train.txt')
        valid_path = os.path.join(self.path, 'dev.txt')
        test_path = os.path.join(self.path, 'test.txt')
        super(Countries, self).__init__(train_path, valid_path, test_path, include_reverse=include_reverse)
