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


# with codecs.open('./data/temp.json', 'w', encoding='utf8') as writer:
#     json.dump(train_data, writer, indent=4, ensure_ascii=False)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, bottleneck_size, growth_rate):
        super(DenseLayer, self).__init__()
        count_of_1x1 = bottleneck_size
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_channels, count_of_1x1, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(count_of_1x1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3x3 = nn.Conv2d(count_of_1x1, growth_rate, kernel_size=3, padding=1)

    def forward(self, *prev_features):
        input = torch.cat(prev_features, dim=1)
        # print(input.device, input.shape)
        out = self.bn1(input)
        out = self.relu1(out)
        bottleneck_output = self.conv1x1(out)
        # bottleneck_output = self.conv1x1(self.relu1(self.bn1(input)))
        out = self.conv3x3(self.relu2(self.bn2(bottleneck_output)))

        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.BertNER = BertModel.from_pretrained(MODEL_PATH, config=model_config)
        self.Dense = DenseLayer(in_channels=1, bottleneck_size=48, growth_rate=12)
        self.Dense2 = DenseLayer(in_channels=12, bottleneck_size=4, growth_rate=1)

    def forward(self, t):
        t = self.BertNER(t).last_hidden_state
        # 对T进行操作，增加一个channel的纬度
        t = t.unsqueeze(1)
        pn1 = self.Dense(t)
        pn1 = self.Dense2(pn1)
        pn2 = self.Dense(t)
        pn2 = self.Dense2(pn2)

        return pn1, pn2


# 实例化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NER = Model().to(device)

optimizer = optim.Adam(NER.parameters(), lr=0.001)


def train_epoch():
    NER.train()
    for batch_index, batch_dict in enumerate(data_generator(train_data, batch_size=10)):
        optimizer.zero_grad()
        # print(len(batch_dict[batch_index][0]))
        t_in = torch.from_numpy(batch_dict[batch_index][0])
        n1 = torch.from_numpy(batch_dict[batch_index][1])
        n2 = torch.from_numpy(batch_dict[batch_index][2])
        pn1, pn2 = NER(t_in)
        pn1 = pn1[:, 0, :, 0]
        pn2 = pn2[:, 0, :, 0]
        # mask = Lambda(lambda x: torch.cast(torch.greater(torch.expand_dims(x, 2), 0), 'float32'))(t_in)
        n1_loss = nn.CrossEntropyLoss()
        n1_loss = n1_loss(n1, pn1)
        # # n1_loss = torch.sum(n1_loss * mask) / torch.sum(mask)
        n2_loss = nn.CrossEntropyLoss()
        n2_loss = n2_loss(n2, pn2)
        # # n2_loss = torch.sum(n2_loss * mask) / torch.sum(mask)
        loss = n1_loss + n2_loss
        # loss.backward()
        # optimizer.step()
        print(loss)
        break


train_epoch()
