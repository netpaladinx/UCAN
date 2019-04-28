import os
import random

class ToyFactoryBase(object):
    sampling_ratios = {}

    def make_dataset(self, data, name, sampling_ratio=1., train_ratio=0.5):
        indices = list(range(len(data)))
        random.shuffle(indices)
        end = int(len(data) * sampling_ratio)
        train_end = int(end * train_ratio)
        valid_end = int((end - train_end) * 0.5) + train_end
        train = [data[i] for i in indices[:train_end]]
        valid = [data[i] for i in indices[train_end:valid_end]]
        test = [data[i] for i in indices[valid_end:end]]

        dir_path = os.path.join('data/toy', name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        with open(os.path.join(dir_path, 'train.txt'), 'w') as fout:
            for line in train:
                fout.write('\t'.join(line) + '\n')

        with open(os.path.join(dir_path, 'valid.txt'), 'w') as fout:
            for line in valid:
                fout.write('\t'.join(line) + '\n')

        with open(os.path.join(dir_path, 'test.txt'), 'w') as fout:
            for line in test:
                fout.write('\t'.join(line) + '\n')

        train_set = set(train)
        valid_set = set(valid)
        test_set = set(test)
        with open(os.path.join(dir_path, 'data.txt'), 'w') as fout:
            for line in data:
                if line in train_set:
                    fout.write('\t'.join(line) + '\ttrain\n')
                if line in valid_set:
                    fout.write('\t'.join(line) + '\tvalid\n')
                if line in test_set:
                    fout.write('\t'.join(line) + '\ttest\n')

        return train, valid, test

    def pick(self, relation):
        return random.random() < self.sampling_ratios[relation]

class Toy1Factory(ToyFactoryBase):
    """ Strict cluster membership.
        { A -> { B }_n }_m
        a - R1 -> b
        b - R1_T -> a
        b1 - R2 -> b2
        a1 - R3 -> a2
    """
    sampling_ratios = {'R1': 0.5, 'R1_T': 0.5, 'R2': 0.5, 'R3': 0.2}

    def __init__(self, n_a, min_n_b, max_n_b):
        data = []
        a_i = 1
        b_i = 1
        for i in range(n_a):
            n_b = random.randint(min_n_b, max_n_b)
            for j in range(n_b):
                a = 'a{:d}'.format(a_i)
                b = 'b{:d}'.format(b_i + j)
                if self.pick('R1'):
                    data.append((a, 'R1', b))
                if self.pick('R1_T'):
                    data.append((b, 'R1_T', a))
            for j1 in range(n_b):
                for j2 in range(n_b):
                    if j1 != j2:
                        b1 = 'b{:d}'.format(b_i + j1)
                        b2 = 'b{:d}'.format(b_i + j2)
                        if self.pick('R2'):
                            data.append((b1, 'R2', b2))
            a_i = a_i + 1
            b_i = b_i + n_b
        a_i = 1
        for i1 in range(n_a):
            for i2 in range(n_a):
                if i1 != i2:
                    a1 = 'a{:d}'.format(a_i + i1)
                    a2 = 'a{:d}'.format(a_i + i2)
                    if self.pick('R3'):
                        data.append((a1, 'R3', a2))

        self.train, self.valid, self.test = self.make_dataset(data, 'toy1')
        print('n_train: {:d} | n_valid: {:d} | n_test: {:d}'.format(len(self.train), len(self.valid), len(self.test)))


class Toy2Factory(ToyFactoryBase):
    """ Tag a instance with multiple classes.
        A: { { A }_n -> B }_m  or  B: { A -> { B }_n }_m
        a - R1 -> b
        b - R1_T -> a
        b1 - R2 -> b2
    """
    pass


class Toy3Factory(ToyFactoryBase):
    """ A -> B -> D
        C -> D
        A: { including all } and { at least one } and { coverage > p% } for B
        B: { including all } and { at least one } and { coverage > p% } for D
        C: { including all } and { at least one } and { coverage > p% } for D
        a - R1 -> b
        b - R1_T -> a
        b - R2 -> d
        d - R2_T -> b
        c - R3 -> d
        d - R3_T -> c
        d1 - R4 -> d2 (for same c)
        d1 - R5 -> d2 (for same b)
        b1 - R6 -> b2 (for same a)
    """
    pass


if __name__ == '__main__':
    #toy1 = Toy1Factory(10, 10, 10)
