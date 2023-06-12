from dataloder2 import DataIndex
from torch.utils.data import DataLoader
from tqdm import tqdm
import codecs
import json

train_data = []
with codecs.open('./data/train_data.json', 'r', encoding='utf8') as reader:
    train_data = json.load(reader)

train = DataIndex(train_data, batch_size=8)


for t, n1, n2 in tqdm(train):
    print(len(t))
