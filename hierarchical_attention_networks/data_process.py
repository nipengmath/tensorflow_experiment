# -*- coding: utf-8 -*-

"""
处理原始数据
"""

import codecs, json, random
from collections import defaultdict
import jieba


def gen_corpus():
    path = "data/sentiment/sentiment.corpus.dat"
    train_path = "data/sentiment/train.dat"
    dev_path = "data/sentiment/dev.dat"

    f_train = codecs.open(train_path, "w")
    f_dev = codecs.open(dev_path, "w")

    z2label = {"1": "2", "0": "0", "-1": "1"}
    ## z2label = {"1": "1", "-1": "0"}

    label_info = defaultdict(list)
    with codecs.open(path) as f:
        for line in f:
            row = json.loads(line)
            z = row["z"]

            if z not in z2label:
                continue

            label = z2label[z]

            x = row["x"]
            x = x.split("_")[0]
            x = x.split("-")[0]
            row["x"] = x
            row["z"] = label
            label_info[label].append(row)

    train_lst = []
    dev_lst = []

    for label, l in label_info.iteritems():
        print(label, len(l))
        for row in l[:10000]:
            rr = random.random()
            if rr > 0.2:
                lst = train_lst
                #f = f_train
            else:
                lst = dev_lst
                #f = f_dev
            x = json.dumps(row, ensure_ascii=False)
            x = x.encode("utf-8")
            lst.append(x)

    print(len(train_lst), len(dev_lst))

    random.shuffle(train_lst)
    random.shuffle(dev_lst)

    for x in train_lst:
        f_train.write(x+"\n")
    for x in dev_lst:
        f_dev.write(x+"\n")

    f_train.close()
    f_dev.close()


def gen_glove_corpus():
    path = "data/sentiment/sentiment.corpus.dat"

    glove_path = "data/glove/corpus.dat"
    fout = codecs.open(glove_path, "w", "utf-8")

    label_info = defaultdict(list)
    with codecs.open(path) as f:
        for line in f:
            row = json.loads(line)
            label = row["z"]
            x = row["x"]
            x = x.split("_")[0]
            x = x.split("-")[0]
            l = list(jieba.cut(x))
            tmp = " ".join(l)
            fout.write(tmp+"\n")
    fout.close()


if __name__ == "__main__":
    print("ok")
    gen_corpus()
    ## gen_glove_corpus()
