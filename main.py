import torch
import logging
import torch.nn as nn
from tqdm import tqdm
# from metrics import f1_score, bad_case
# from DenseLayer import DenseNet
from transformers import BertTokenizer, BertModel, BertConfig
from dataloder import data_generator

MODEL_PATH = './guwenbert-base/'

model_config = BertConfig.from_pretrained("ethanyt/guwenbert-base")
model_config.output_hidden_states = True
model_config.output_attentions = True
model_config = BertConfig.from_pretrained("ethanyt/guwenbert-base")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.BertNER = BertModel.from_pretrained(MODEL_PATH, config=model_config)
        self.Dense = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

    def forward(self, t1, t2, n1, n2):
        mask = Lambda(lambda x: torch.cast(torch.greater(torch.expand_dims(x, 2), 0), 'float32'))(t1)
        t = self.BertNER([t1, t2])
        pn1 = self.Dense(t)
        pn2 = self.Dense(t)
        n1_loss = nn.CrossEntropyLoss(n1, pn1)
        n1_loss = torch.sum(n1_loss * mask) / torch.sum(mask)
        n2_loss = nn.CrossEntropyLoss(n2, pn2)
        n2_loss = torch.sum(n2_loss * mask) / torch.sum(mask)
        loss = n1_loss + n2_loss
        return (loss,) + (pn1, pn2)

# def train_epoch():
#     Model.train()
#     train_losses = 0
#     for t1, t2, n1, n2 in enumerate(tqdm(data_generator)):
#         loss = Model([t1, t2])
#         train_losses += loss.item()
#         Model.zero_grad()
#         loss.backward()

print(Model())