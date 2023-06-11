import json
import codecs

chars = {}
train_data = []

with codecs.open('./../data/dev_data.json', 'r', encoding='utf8') as f:
    for line in f.readlines():
        ner = []
        a = json.loads(line)
        text = a['text']
        for i in a['label']:
            ner.append(text[i[0]:i[1]])
        train_data.append(
            {
                'text': text,
                'ner': ner
            }
        )
        for c in a['text']:
            chars[c] = chars.get(c, 0) + 1
with codecs.open('./../data/dev_data_clean.json', 'w', encoding='utf8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)
    f.close()

# with codecs.open('./../data/all_chars.json', 'w', encoding='utf8') as f:
#     chars = {i:j for i,j in chars.items() if j >= 2}
#     id2char = {i+2:j for i,j in enumerate(chars)}
#     char2id = {j:i for i,j in id2char.items()}
#     json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)
#     f.close()