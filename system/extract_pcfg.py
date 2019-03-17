# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:07:42 2019
@author: juliette rengot

Module to extract a PCFG
"""
#####################################################################
#                        IMPORT MODULES                             #
#####################################################################
import codecs
import pickle
from collections import defaultdict
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
import matplotlib.pyplot as plt
    
def pcfg_extraction():
    #####################################################################
    #                      Load Corpus treebanks                        #
    #####################################################################
    treebank_train = BracketParseCorpusReader("", "sequoia_train.tb")
    
    #####################################################################
    #    Initialisation for unary, binary and terminal rules            #
    #####################################################################
    #Unary rules
    unary_freq         = defaultdict(float)  # How frequent is the rule A->B ?
    unary_cnt_by_lhs   = defaultdict(int)    # How many times A is the left part of a unary rule ?
    unary_occur_cnt    = 0                   # How many unary rules are there ?
    unary_lhs_set      = set()               # Set of left part symbols
    unary_rhs_set      = set()               # Set of right part symbols
    
    #binary rules
    binary_freq        = defaultdict(float)
    binary_cnt_by_lhs  = defaultdict(int)
    binary_occur_cnt   = 0
    binary_lhs_set     = set()
    binary_rhs_set     = set()
    
    #terminal rules
    postags_freq       = defaultdict(float)
    postags_cnt_by_pos = defaultdict(int)
    postags_occur_cnt  = 0
    words_occur_cnt    = defaultdict(int)
    postags_set        = set()
    words_set          = set()
    
    #####################################################################
    #           Parsing collection of rules and words                   #
    #####################################################################
    for tree in treebank_train.parsed_sents():
        t = tree.copy()
        t.chomsky_normal_form(horzMarkov=2)    # Convert a tree into its Chomsky Normal Form equivalent
        prods = t.productions()                # Get the recursive productions
    
        for prod in prods:
            left_symbol = prod.lhs().symbol()  # Left hand side
            right_part  = prod.rhs()           # Right hand side
    
            if isinstance(right_part[0], str): # Termination found : left side = part-of-speech tags
                right_symbol = right_part[0]
                #save it in terminal rules
                postags_freq[(left_symbol, right_symbol)] += 1
                postags_cnt_by_pos[left_symbol] += 1
                postags_occur_cnt += 1
                words_occur_cnt[right_symbol] += 1
                postags_set.add(left_symbol)
                words_set.add(right_symbol)
                
            else:
                if len(right_part) == 1:       # Unary found
                    right_symbol = right_part[0].symbol()
                    #save it in unary rules
                    unary_freq[(left_symbol, right_symbol)] += 1
                    unary_cnt_by_lhs[left_symbol] += 1
                    unary_occur_cnt += 1
                    unary_lhs_set.add(left_symbol)
                    unary_rhs_set.add(right_symbol)
                    
                elif len(right_part) == 2:     # Binary found
                    right_symbol = tuple([nt.symbol() for nt in right_part])
                    #save it in binary rules
                    binary_freq[(left_symbol, right_symbol)] += 1
                    binary_cnt_by_lhs[left_symbol] += 1
                    binary_occur_cnt += 1
                    binary_lhs_set.add(left_symbol)
                    binary_rhs_set.add(right_symbol)
                    
    #####################################################################
    #           Look at the occurences of part-of-speech tags           #
    #####################################################################
    n_tag = len(words_occur_cnt.keys())
    print('There are ' + str(n_tag) +
          ' different part-of-speech tags in the training set')
    plt.scatter([i for i in range(len(words_occur_cnt.keys()))],
              [words_occur_cnt[i] for i in words_occur_cnt.keys()])
    plt.title('Occurences of part-of-speech tags')
    plt.xlabel('tag')
    plt.ylabel('occurence')
    plt.show()
    #####################################################################
    #             Group rare words into a new tag UNK                   #
    #####################################################################
    # Replace rare words with '<UNKNOWN>' tag
    unfrequent = set([w for w in words_set if words_occur_cnt[w] < 2])
    T_set = words_set.copy()
    T_set.difference_update(unfrequent)
    T_set.add(u"<UNKNOWN>")
    pw_pairs = list(postags_freq.keys())
    for (pos, w) in pw_pairs:
        if w in unfrequent:
            postags_freq[(pos, u"<UNKNOWN>")] += postags_freq[(pos, w)]
            postags_freq.pop((pos, w))
    
    #####################################################################
    #                          Normalisation                            #
    #####################################################################
    for (pos, w) in postags_freq:
        postags_freq[(pos, w)] /= postags_cnt_by_pos[pos]
    
    for (lhs, rhs) in unary_freq:
        unary_freq[(lhs, rhs)] /= (unary_cnt_by_lhs[lhs] + binary_cnt_by_lhs[lhs])
        
    for (lhs, rhs) in binary_freq:
        binary_freq[(lhs, rhs)] /= (unary_cnt_by_lhs[lhs] + binary_cnt_by_lhs[lhs])
    
    #####################################################################
    #                   Save the results in files                       #
    #####################################################################
    with codecs.open("PCFG_unary_freq.pkl", 'wb') as file:
        pickle.dump(unary_freq, file)
    file.close()
    
    with codecs.open("PCFG_binary_freq.pkl", 'wb') as file:
        pickle.dump(binary_freq, file)
    file.close()
    
    with codecs.open("PCFG_postags_freq.pkl", 'wb') as file:
        pickle.dump(postags_freq, file)
    file.close()
    
    #####################################################################
    #                       rhs -> lhs dictionary                       #
    #####################################################################
    unary_dict  = {}
    binary_dict = {}
    postags_dict      = {}
    
    for rhs in unary_rhs_set:
        unary_dict[rhs] = {}
    for (lhs, rhs) in unary_freq:
        unary_dict[rhs][lhs] = unary_freq[(lhs, rhs)]
        
    for rhs in binary_rhs_set:
        binary_dict[rhs] = {}
    for (lhs, rhs) in binary_freq:
        binary_dict[rhs][lhs] = binary_freq[(lhs, rhs)]
        
    for w in T_set:
        postags_dict[w] = {}
    for (pos, w) in postags_freq:
        postags_dict[w][pos] = postags_freq[(pos, w)]
    
    #####################################################################
    #                   Save the results in files                       #
    #####################################################################
    with codecs.open("PCFG_unary_dict.pkl", 'wb') as file:
        pickle.dump(unary_dict, file)
    file.close()
    
    with codecs.open("PCFG_binary_dict.pkl", 'wb') as file:
        pickle.dump(binary_dict, file)
    file.close()
    
    with codecs.open("PCFG_postags_dict.pkl", 'wb') as file:
        pickle.dump(postags_dict, file)
    file.close()
    
    #####################################################################
    #            the set of non-terminals and terminals                 #
    #####################################################################
    # Store the set of non-terminals and terminals
    NT_set = unary_lhs_set.union(binary_lhs_set)
    
    with codecs.open("NT_set.pkl", 'wb') as file:
        pickle.dump(NT_set, file)
    file.close()
    
    with codecs.open("T_set.pkl", 'wb') as file:
        pickle.dump(T_set, file)
    file.close()
    
    with codecs.open("postags_set.pkl", 'wb') as file:
        pickle.dump(postags_set, file)
    file.close()
    
    with codecs.open("words_set.pkl", 'wb') as file:
        pickle.dump(words_set, file)
    file.close()
    
    return()

if __name__ == '__main__' :
    pcfg_extraction()