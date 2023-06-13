import numpy as np
import os
import torch
import codecs
import json
import logging
import torch.nn as nn
from tqdm import tqdm
from torchvision.transforms import Lambda
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertConfig
from dataloder import DataIndex
import time
import torch.distributed as dist
from Modules.Email import Email

# Notice
mail = Email(receivers="guoqingzhang@kuhp.kyoto-u.ac.jp")

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5'
# dist.init_process_group(backend='nccl')

now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

MODEL_PATH = 'ethanyt/guwenbert-base'
Ourtokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model_config = BertConfig.from_pretrained(MODEL_PATH)

dev_data = []
with codecs.open('./data/dev_data_clean.json', 'r', encoding='utf8') as reader:
    dev_data = json.load(reader)

train_data = []
with codecs.open('./data/train_data.json', 'r', encoding='utf8') as reader:
    train_data = json.load(reader)
    reader.close()

# with codecs.open('./data/temp.json', 'w', encoding='utf8') as writer:
#     json.dump(train_data, writer, indent=4, ensure_ascii=False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.BertNER = BertModel.from_pretrained(MODEL_PATH, config=model_config)
        self.Liner = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t):
        t = self.BertNER(t).last_hidden_state
        pn1 = self.Liner(t)
        pn1 = self.sigmoid(pn1)
        pn2 = self.Liner(t)
        pn2 = self.sigmoid(pn2)

        return pn1, pn2


def train_epoch():
    NER.train()
    for t_in, n1, n2 in tqdm(DataIndex(train_data, batch_size=16)):
        optimizer.zero_grad()
        t_in = torch.as_tensor(t_in, device=torch.device('cuda'))
        n1 = torch.as_tensor(n1, device=torch.device('cuda'))
        n2 = torch.as_tensor(n2, device=torch.device('cuda'))
        n1 = torch.unsqueeze(n1, 2)
        n2 = torch.unsqueeze(n2, 2)
        pn1, pn2 = NER(t_in)
        # mask = Lambda(lambda x: torch.cast(torch.greater(torch.expand_dims(x, 2), 0), 'float32'))(t_in)
        n1_loss = nn.BCEWithLogitsLoss()
        n1_loss = n1_loss(n1, pn1)
        # n1_loss = torch.sum(n1_loss * mask) / torch.sum(mask)
        n2_loss = nn.BCEWithLogitsLoss()
        n2_loss = n2_loss(n2, pn2)
        # n2_loss = torch.sum(n2_loss * mask) / torch.sum(mask)
        loss = (n1_loss + n2_loss) / 2
        loss.backward()
        optimizer.step()
    return 'loss:%.4f\n' % loss


def extract_items(text_in):
    _t = Ourtokenizer.encode(text_in)
    _t = torch.as_tensor([_t], device=torch.device('cuda'))
    _pn1, _pn2 = NER(_t)
    _pn1 = _pn1.cpu()
    _pn2 = _pn2.cpu()
    _pn1 = _pn1.detach().numpy()
    _pn2 = _pn2.detach().numpy()
    _pn1, _pn2 = np.where(_pn1[0] > 0.7)[0], np.where(_pn2[0] > 0.6)[0]
    _NER = set()
    for i in _pn1:
        j = _pn2[_pn2 >= i]
        if len(j) > 0:
            j = j[0]
            _NERs = text_in[i - 1:j]
            _NER.add(_NERs)
    if _NER:
        return _NER
    else:
        return set()


def evaluate():
    NEREval = torch.load('NER_best.pth')
    NEREval.eval()
    A, B, C = 1e-10, 1e-10, 1e-10
    F = open('./data/eval.csv', 'w', encoding='utf-8')
    for d in dev_data:
        R = extract_items(d['text'][:512])
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
    msg = 'f1: %.4f, precision: %.4f, recall: %.4f\n' % (2 * A / (B + C), A / B, A / C)
    return msg


if __name__ == '__main__':
    device = torch.device('cuda')
    # NER = nn.parallel.DistributedDataParallel(NER, find_unused_parameters=True)
    NER = Model().to(device)
    NER = nn.DataParallel(NER)
    optimizer = optim.Adam(NER.parameters(), lr=2e-5)
    epoch = 10
    final_loss = ''
    for i in range(epoch):
        print("The " + str(i) + " epoch.")
        print(train_epoch())
        if i == epoch-1:
            final_loss = train_epoch()
    torch.save(NER, 'NER_best.pth')
    message = evaluate()
    print(message)
    mail.generator(text='loss:' + str(final_loss) + '\n' + 'eval:' + message, subject='Training Result:' + now)
    mail.send()

