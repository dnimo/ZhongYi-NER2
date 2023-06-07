import json
import codecs

class Openfile():
    def __init__(self):
        pass
    def __call__(self, Path):
        self.Path = Path
        sequence = []
        with codecs.open(Path, 'r',encoding='utf-8') as f:
            for seq in f:
                a = json.loads(seq)
                sequence.append(
                    a['text']
                )
        return sequence

def main():
    path = "//DataLoader/complete.txt"
    openfile = Openfile()
    a = openfile(path)
    print(type(a))
    with codecs.open('complete.json', 'w', encoding='utf-8') as f:
        for i in a:
            f.write(i)
            f.write('\n')


if __name__ == "__main__":
    main()
else:
    print("Error")