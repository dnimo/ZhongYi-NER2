import numpy as np
import torch
import codecs
import json
import logging
import torch.nn as nn
from tqdm import tqdm
# from metrics import f1_score, bad_case
from torchvision.transforms import Lambda
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertConfig
from dataloder import data_generator
import time

now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Ourtokenizer = BertTokenizer.from_pretrained("ethanyt/guwenbert-base")
device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = './guwenbert-base/'

model_config = BertConfig.from_pretrained("guwenbert-base")
# model_config.output_hidden_states = True
# model_config.output_attentions = True

# add guwenBert dict

# dict_path = './guwenbert-base/vocab.txt'
#
# token_dict = {}
#
# with codecs.open(dict_path, 'r', 'utf8') as reader:
#     for line in reader:
#         token = line.strip()
#         token_dict[token] = len(token_dict)


train_data = []
with codecs.open('./data/train_data.json', 'r', encoding='utf8') as reader:
    train_data = json.load(reader)

dev_data = []
with codecs.open('', 'r', encoding='utf8') as reader:
    dev_data = json.load(reader)


# with codecs.open('./data/temp.json', 'w', encoding='utf8') as writer:
#     json.dump(train_data, writer, indent=4, ensure_ascii=False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.BertNER = BertModel.from_pretrained(MODEL_PATH, config=model_config)
        self.Liner = nn.Linear(768, 1)

    def forward(self, t):
        t = self.BertNER(t).last_hidden_state
        # 对T进行操作，增加一个channel的纬度
        pn1 = self.Liner(t)
        pn2 = self.Liner(t)

        return pn1, pn2


# 实例化模型

NER = Model().to(device)

optimizer = optim.Adam(NER.parameters(), lr=2e-5)


def train_epoch():
    NER.train()
    for batch_index, batch_dict in enumerate(data_generator(train_data, batch_size=8)):
        optimizer.zero_grad()
        # print(len(batch_dict[batch_index][0]))
        t_in = torch.from_numpy(batch_dict[0]).to(device)
        n1 = torch.from_numpy(batch_dict[1]).to(device)
        n2 = torch.from_numpy(batch_dict[2]).to(device)
        n1 = torch.unsqueeze(n1, 2)
        n2 = torch.unsqueeze(n2, 2)
        pn1, pn2 = NER(t_in)
        # mask = Lambda(lambda x: torch.cast(torch.greater(torch.expand_dims(x, 2), 0), 'float32'))(t_in)
        n1_loss = nn.BCEWithLogitsLoss()
        n1_loss = n1_loss(n1, pn1)
        # n1_loss = torch.sum(n1_loss * mask) / torch.sum(mask)
        n2_loss = nn.CrossEntropyLoss()
        n2_loss = n2_loss(n2, pn2)
        # n2_loss = torch.sum(n2_loss * mask) / torch.sum(mask)
        loss = n1_loss + n2_loss
        loss.backward()
        optimizer.step()
        print(loss)
    torch.save(NER, 'NER_' + now + '.pth')


# train_epoch()

def extract_items(text_in):
    _token = Ourtokenizer.tokenize(text_in)
    _t = Ourtokenizer.encode(first=text_in)
    _t = np.array([_t])
    _pn1, _pn2 = NER(_t)
    _pn1, _pn2 = np.where(_pn1[0] > 0.5)[0], np.where(_pn2[0] > 0.4)[0]
    _NER = set()
    for i in _pn1:
        j = _pn2[_pn2 >= i]
        if len(j) > 0:
            j = j[0]
            _NERs = text_in[i - 1: j]
            _NER.add(_NERs)
    if _NER:
        return _NER
    else:
        return []


def evaluate():
    NER.eval()
    A, B, C = 1e-10, 1e-10, 1e-10
    F = open('', 'w', encoding='utf-8')
    for d in dev_data:
        R = extract_items(d['text'])
        T = set()
        for iterm in d['ner']:
            T.add(iterm)
        A += len(R & T)
        B += len(R)
        C += len(T)
        ner = json.dumps(
            {
                'text': d['text'],
                'ner_list': [i for i in T],
                'ner_list_pred': [i for i in R],
                'new': [i for i in R - T],
                'lack': [i for i in T - R]
            }, ensure_ascii=False, indent=4
        )
        F.write(ner + '\n')
    F.close()
    print('f1: %.4f, precision: %.4f, recall: %.4f\n' % (2 * A / (B + C), A / B, A / C))
    # return 2 * A / (B + C), A / B, A / C


if __name__ == '__main__':
    train_epoch()
    evaluate()
