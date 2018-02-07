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
    #print (tp, fp, fn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = (2 * p * r) / (p + r)
    score = "p = " + str(p) + ", r = " + str(r) + ", f = " + str(f)
    return score

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
    #print (tp, fp, fn)
    
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = (2 * p * r) / (p + r)
    score = "p = " + str(p) + ", r = " + str(r) + ", f = " + str(f)
    return score

#### take nbest file and make a dictionary to calcualte oracle score
def nbest_oracle(nbest_file):
    pred_nbest = {}

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

#### load MT file ## French to English
def load_mt(fname, mt_pairs, mt_idx):
    fin = open (fname, 'r', encoding='utf-8')
    count = 1
    for line in fin:
        fr, en = line.split("\t")
        mt_pairs[fr] = en
        mt_idx[count] = fr
        count += 1
    return


################# BLEU #######################
def compareNbest_with_mt_bleu(fr, en, mt_pairs):
	from nltk import translate
	mt_en = mt_pairs[fr] #get corresponding translation
	return translate.bleu_score.sentence_bleu([mt_en], en)

#### take nbest file and add an extra column with bleu score
def nbest_bleu_save(in_file, out_file, mt_pairs):
    
    nbest_out = open (out_file, 'w', encoding='utf-8')
    nbest = open (in_file, 'r', encoding='utf-8')

    curr_bleu = 0
    for line in nbest:
        line = line.rstrip()
        fr_id, fr, en_id, en = line.split("\t")

        curr_bleu = compareNbest_with_mt_bleu(fr, en, mt_pairs)

        nbest_out.write (line + "\t" + str(curr_bleu) + "\n")
    nbest_out.close()


### load nbest file with bleu and choose the candidate with best bleu
def getBestBleu(file_train_nbest_bleu):
    import sys
    nbest_bleu = open (file_train_nbest_bleu, 'r', encoding='utf-8')

    best_bleu = -1
    best_str = ""
    dic = {}

    for line in nbest_bleu:
        line = line.rstrip()
        fr_id, fr, en_id, en, bleu = line.split("\t")

        if fr_id not in dic:
            dic[fr_id] = line
            best_bleu = bleu
            best_str = line
        elif bleu > best_bleu:
            best_bleu = bleu
            best_str = line
            dic[fr_id] = best_str
    return dic

### get best bleu file and threshold on bleu value
def filter_on_bleu(dic, threshold):
    final_list = {}
    eval_list = {}
    for key,value in dic.items():
        fr_id, fr, en_id, en, bleu = value.split("\t")
        if float(bleu) > threshold:
            final_list[key] = value
            fr_id = str(int(fr_id.replace('fr-','')))
            en_id = str(int(en_id.replace('en-','')))
            eval_list[fr_id + "\t" + en_id] = float(bleu)
    return eval_list

######### TER ###################
def compareNbest_with_mt_ter(fr, en, mt_pairs):
	import pyter
	mt_en = mt_pairs[fr].split() #get corresponding translation
	en_wrd = en.split()
	return pyter.ter(en_wrd, mt_en)

#### take nbest file and add an extra column with bleu score

def nbest_ter_save(in_file, out_file, mt_pairs):
    
    nbest_out = open (out_file, 'w', encoding='utf-8')
    nbest = open (in_file, 'r', encoding='utf-8')

    curr_ter = 0
    for line in nbest:
        line = line.rstrip()
        fr_id, fr, en_id, en = line.split("\t")

        curr_ter = compareNbest_with_mt_ter(fr, en, mt_pairs)

        nbest_out.write (line + "\t" + str(curr_ter) + "\n")
    nbest_out.close()

