import json
import os
import numpy as np
from transformers import BertTokenizer

maxlen = 512

Ourtokenizr = BertTokenizer.from_pretrained("ethanyt/guwenbert-base")


def list_find(list1, list2):
    n_list2 = len(list2)
    for i in list(range(len(list1))):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array(
        [
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ]
    )


# if not os.path.exists('./data/random_order_train.json'):
#     random_order = list(range(len(train_data)))
#     np.random.shuffle(random_order)
#     json.dump(
#         random_order,
#         codecs.open('./data/random_order_train.json', 'w', encoding='utf8'),
#         indent=4,
#         ensure_ascii=False
#     )
# else:
#     random_order = json.loads(codecs.open('/data/random_order_train.json', 'r', encoding='utf8'))

class data_generator:
    def __init__(self, data, batch_size=10):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T1, T2, N1, N2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d['text'][:maxlen]
                print(type(text))
                tokens = Ourtokenizr.tokenize(text)
                Setner = []
                keys = []
                for ner in d['ner']:
                    Setner.append(Ourtokenizr.tokenize(ner))
                for nner in Setner:
                    Nerid = list_find(tokens, nner)
                    if Nerid != -1:
                        keys.append([Nerid, Nerid + len(nner)])
                        # if key not in items:
                        #     items[key] = []
                        # items[key].append()
                t = Ourtokenizr.encode(text)
                T1.append(t)
                T2.append(t)
                n1, n2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                for j in keys:
                    n1[j[0]] = 1
                    n2[j[1] - 1] = 1
                N1.append(n1)
                N2.append(n2)
                if len(T1) == self.batch_size or i == idxs[-1]:
                    T1 = seq_padding(T1)
                    T2 = seq_padding(T2)
                    N1 = seq_padding(N1)
                    N2 = seq_padding(N2)
                    yield [T1, T2, N1, N2], None
                    T1, T2, N1, N2 = [], [], [], []
