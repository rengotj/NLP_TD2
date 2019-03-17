# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:51:38 2019
@author: juliette rengot

A class for  Cocke–Younger–Kasami (CYK) parser
"""

#####################################################################
#                        IMPORT MODULES                             #
#####################################################################
import codecs
from nltk import Tree
from collections import defaultdict

#####################################################################
#                             CYK parsing                           #
#####################################################################
class CYK_parser(object):
    def __init__(self):
        self.NT_set            = set()               # set of non-terminal symbols
        self.T_set             = set()               # set of terminal symbols
        self.postags_set       = set()               # set of postags
        self.unary_freq        = defaultdict(float)  # frequencies of unary rules (A -> B)
        self.binary_freq       = defaultdict(float)  # frequencies of binary rules (A -> BC)
        self.postags_freq      = defaultdict(float)  # frequencies of postags (POS -> <word>)
        self.unary_dict        = {}
        self.binary_dict       = {}
        self.postags_dict      = {}
        self.not_initialized   = True

    def initialize(self, NT_set, T_set, postags_set,
                   unary_freq, binary_freq, postags_freq,
                   unary_dict, binary_dict, postags_dict):
        self.NT_set            = NT_set
        self.T_set             = T_set
        self.postags_set       = postags_set
        self.unary_freq        = unary_freq
        self.binary_freq       = binary_freq
        self.postags_freq      = postags_freq
        self.unary_dict        = unary_dict
        self.binary_dict       = binary_dict
        self.postags_dict      = postags_dict
        self.not_initialized   = False
        
    def parsing(self, s):
        token_list = s.strip().split(u' ')
        n = len(token_list)
        dp = defaultdict(float)
        backward = {}
        
        for i, w in enumerate(token_list):
            if w in self.T_set:
                dp[(i, i+1)] = self.postags_dict[w]
            else:
                dp[(i, i+1)] = self.postags_dict[u"<UNKNOWN>"]
            self.add_unary(dp, backward, i, i+1)
        
        for l in range(2, n + 1):         # Len of span
            for i in range(0, n + 1 - l): # Start of span
                j = i + l
                dp[(i, j)] = {}
                
                for s in range(i + 1, j): # Partition of span
                    B_set = dp[(i, s)]
                    C_set = dp[(s, j)]
                    
                    for B in B_set.keys(): #For each production A -> B, C
                        prob_B = B_set[B]
                        
                        for C in C_set.keys():
                            prob_C = C_set[C]
                            if (B, C) in self.binary_dict:
                            
                                for A in self.binary_dict[(B, C)].keys():
                                    prob_A = self.binary_dict[(B, C)][A]
                                    prob = prob_A * prob_B * prob_C
                                    if (A not in dp[(i, j)]) or prob > dp[(i, j)][A]:
                                        #update
                                        dp[(i, j)][A] = prob
                                        backward[(i, j, A)] = (s, B, C)
                                        
                self.add_unary(dp, backward, i, j)
        
        if (0, n, u"SENT") not in backward:
            return(None)
        
        else:
            t = self.tree_construction(backward, 0, n, u"SENT", token_list)
            t.un_chomsky_normal_form(expandUnary=False)
            return(t)
 
    def add_unary(self, dp, backward, i, j):
        B_set = list(dp[(i, j)].keys()).copy()
        for B in B_set:
            if B in self.unary_dict:
                for A in self.unary_dict[B].keys(): #for each production A->B
                    prob_A = self.unary_dict[B][A]
                    prob = prob_A * dp[(i, j)][B]
                    if (A not in dp[(i, j)]) or prob>dp[(i, j)][A]:
                        #update
                        dp[(i, j)][A] = prob
                        backward[(i, j, A)] = (B,)
        return()
    
    def tree_construction(self, backward, i, j, label, token_list):
        if (i, j, label) not in backward: # Terminals
            t = Tree(label, [token_list[i]])
        elif len(backward[(i, j, label)]) == 1: # Unary rules
            child_label = backward[(i, j, label)][0]
            t = Tree(label, [self.tree_construction(backward, i, j, child_label, token_list)])
        else: # Binary rules
            split, child_label0, child_label1 = backward[(i, j, label)]
            t = Tree(label, [self.tree_construction(backward, i, split, child_label0, token_list),
                             self.tree_construction(backward, split, j, child_label1, token_list)])
        return(t)

    ### main functions ###
    def parse_sentence(self, input, output=None):
        if self.not_initialized:
            print("Need of initialisation")
            return

        tree = self.parsing(input)
        if output == None:
            print(tree)
        else:
            with codecs.open(output, 'w', 'UTF-8') as f:
                f.write(u"{0}\n".format(tree))
                f.close()
        return(tree)
        
        
    def parse_corpus(self, input, output=None):
        if self.not_initialized:
            print("Need of initialisation")
            return

        with codecs.open(input, 'r', 'UTF-8') as f_in:
            if output != None:
                f_out = codecs.open(output, 'w', 'UTF-8')
            data = f_in.read().splitlines()
            print(len(data))
            for sent in data:
                tree = self.parsing(sent)
                if output != None:
                    f_out.write(u"{0}\n".format(tree))
                else:
                    print(u"{0}\n".format(tree))

            f_in.close()
            if output != None:
                f_out.close()
        return()