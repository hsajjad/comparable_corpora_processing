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

# take an nbest input file and get the best possible score
def calculateOracleScore(pred, gold, nbest):
    tp = 0
    fp = 0
    fn = 0

    for key,value in pred.items():
        if key in gold:
            gold[key] = 1
            tp += 1
            pred[key] = 1

    fp = len(pred)/nbest - tp ## normlalize # predicted minus gold = ones that are not in gold
    fn = len(gold) - tp # present in gold but not in predicted
    print (tp, fp, fn)
     
    
    print (tp, fp, fn)
    
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = (2 * p * r) / (p + r)
    print (p, r, f)

#### take nbest file and make a dictionary to calcualte oracle score
def nbest_oracle(nbest_file, pred_nbest):
    
    nbest = open (nbest_file, 'r', encoding='utf-8')
    for line in nbest:
        line = line.rstrip()
        fr_id, fr, en_id, en = line.split("\t")
        fr_id = str(int(fr_id.replace('fr-','')))
        en_id = str(int(en_id.replace('en-','')))
        pred_nbest[fr_id + "\t" + en_id] = 0
    
    nbest.close()
    return pred_nbest


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
