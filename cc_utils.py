import gensim
import sklearn
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy


def calculateScore(pred, gold):
    tp = 0
    fp = 0
    fn = 0

    for key,value in pred.items():
        if key in gold:
            gold[key] = 1
            tp += 1
            pred[key] = 1

    fp = len(pred) - tp # predicted minus gold = ones that are not in gold
    fn = len(gold) - tp # present in gold but not in predicted
    print (tp, fp, fn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = (2 * p * r) / (p + r)
    print (p, r, f)


def load_gold(fname):
    fin = open (fname, 'r', encoding='utf-8')
    gold = {}
    for line in fin:
        fr, en = line.split("\t")
        fr_int = int(fr.replace('fr-',''))
        en_int = int(en.replace('en-',''))
    
        gold[str(fr_int) + "\t" + str(en_int)] = 0 # making french english pairs
    return (gold)

def load_prediction(fname):
    fin = open (fname, 'r', encoding='utf-8')
    pred = {}
    for line in fin:
        line = line.rstrip()
        pred[line] = 0 # loading french english pairs
    return (pred)
