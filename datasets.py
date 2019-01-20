import os
import csv
import json
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445
NLI_LABELS = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

def _rocstories(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y

def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)

def _snli(path):
    sents1 = []
    sents2 = []
    labels = []
    with open(path, encoding='utf_8') as f:
        seen_header = False
        for line in f:
            if not seen_header:
                seen_header = True
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            fields = line.split('\t')
            label_txt = fields[0]
            if label_txt == '-':
                continue
            label = NLI_LABELS[label_txt]
            s1 = fields[5]
            s2 = fields[6]
            sents1.append(s1)
            sents2.append(s2)
            labels.append(label)
        return sents1, sents2, labels

def snli(data_dir):
    train_sents1, train_sents2, train_ys = _snli(os.path.join(data_dir, 'snli_1.0_train.txt'))
    dev_sents1, dev_sents2, dev_ys = _snli(os.path.join(data_dir, 'snli_1.0_dev.txt'))
    test_sents1, test_sents2, _ = _snli(os.path.join(data_dir, 'snli_1.0_test.txt'))
    train_ys = np.asarray(train_ys, dtype=np.int32)
    dev_ys = np.asarray(dev_ys, dtype=np.int32)
    return (train_sents1, train_sents2, train_ys), (dev_sents1, dev_sents2, dev_ys), (test_sents1, test_sents2)

def _mednli(path):
    sents1 = []
    sents2 = []
    labels = []
    with open(path, encoding='utf_8') as f:
        for line in f:
            json_obj = json.loads(line)
            sents1.append(json_obj['sentence1'])
            sents2.append(json_obj['sentence2'])
            labels.append(NLI_LABELS[json_obj['gold_label']])

        return sents1, sents2, labels

def mednli(data_dir):
    train_sents1, train_sents2, train_ys = _mednli(os.path.join(data_dir, 'mli_train_v1.jsonl'))
    dev_sents1, dev_sents2, dev_ys = _mednli(os.path.join(data_dir, 'mli_dev_v1.jsonl'))
    test_sents1, test_sents2, _ = _mednli(os.path.join(data_dir, 'mli_test_v1.jsonl'))
    train_ys = np.asarray(train_ys, dtype=np.int32)
    dev_ys = np.asarray(dev_ys, dtype=np.int32)
    return (train_sents1, train_sents2, train_ys), (dev_sents1, dev_sents2, dev_ys), (test_sents1, test_sents2)
