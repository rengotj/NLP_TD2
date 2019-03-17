# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:01:14 2019

@author: juliette rengot
A module for evaluation of obtain parsing
"""
#####################################################################
#                        IMPORT MODULES                             #
#####################################################################
import codecs
import numpy as np
from PYEVALB import scorer, parser

def evaluation():
    #####################################################################
    #                              Load data                            #
    #####################################################################
    with codecs.open("output.txt", 'r', 'UTF-8') as file:
        result = file.read()
    file.close()
    result = result.split()
    result_tree = []
    i=-1
    for r in result:
        if 'None' in r :
            result_tree.append('(SENT (NC <UNKNOWN>))')
            i += 1
        elif 'SENT' in r :
            result_tree.append(r)
            i += 1
        else :
            result_tree[i] = result_tree[i] + ' ' + r

    with codecs.open("sequoia_test_tree.txt", 'r', 'UTF-8') as file:
        truth = file.read()
    file.close()
    truth = truth.split()
    truth_tree = []
    i=-1
    for t in truth:
        if 'SENT' in t:
            truth_tree.append(t)
            i += 1
        else :
            truth_tree[i] = truth_tree[i] + ' ' + t
    
    assert(len(result_tree)==len(truth_tree))
    N = len(result_tree)
    
    #####################################################################
    #                            Evaluation                             #
    #####################################################################
    recall = []
    precision = []
    Fscore=[]
    tag_accuracy=[]
    
    S = scorer.Scorer()
    fileOut = codecs.open("evaluation_data.parser_output", 'w', 'UTF-8')
    
    for i in range(N):
        t = parser.create_from_bracket_string(truth_tree[i])
        r = parser.create_from_bracket_string(result_tree[i])
        
        fileOut.write(" ".join(str(t.non_terminal_labels)))
        fileOut.write('\n')
        
        if t.sentence == r.sentence :
            scores = S.score_trees(t, r)
            recall.append(scores.recall)
            precision.append(scores.prec)
            Fscore.append(2*scores.recall*scores.prec/(scores.prec+scores.recall))
            tag_accuracy.append(scores.tag_accracy)
    
    print('Average recall : ', np.mean(recall))
    print('Average precision : ', np.mean(precision))
    print('Average F-score: ', np.mean(Fscore))
    print('Average tag accuracy: ', np.mean(tag_accuracy))

    return()

if __name__ == '__main__' :
    evaluation()