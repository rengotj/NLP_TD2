# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:02:34 2019
@author: juliette rengot
Main function for TD2
"""
#####################################################################
#                        IMPORT MODULES                             #
#####################################################################
import os
import codecs
import pickle
import argparse
import sys
sys.path.append(sys.argv[8])
import CYK_parser_class

#####################################################################
#                        Fonction arguments                         #
#####################################################################
"""
argparser = argparse.ArgumentParser()
argparser.add_argument('--use_prepare_data', type=bool, default=True) # Needed for the first use
argparser.add_argument('--use_extract_pcfg', type=bool, default=True) # Needed for the first use
argparser.add_argument('--use_OOV', type=bool, default=True)          # Needed for Out Of Vocabulary handling
argparser.add_argument('--use_OOV_reversal', type=bool, default=True) # Needed if use_OOV and if we accept swap
                                                                      # of adjacent character (Damerau extension)
argparser.add_argument('--use_OOV_prob', type=bool, default=True)     # Needed if use_OOV and if we compute
                                                                      # probability for cost estimation
argparser.add_argument('--use_OOV_context', type=bool, default=True)  # Needed if use_OOV and use_OOV_prob_use and
                                                                      # if we take into account context in probability computation
argparser.add_argument('--use_eval', type=bool, default=True)         # Needed for precision, recall, fscore and
                                                                      # tag accuracy computation
argparser.add_argument('--current_folder', type=str, default='')      # path to find the files

args = argparser.parse_args()
"""

#####################################################################
#                  Files construction if needed                     #
#####################################################################
if sys.argv[1] == 'True' :
    import prepare_data
    prepare_data.data_preparation()
    print('prepare data Done')
if sys.argv[2] == 'True':
    import extract_pcfg
    extract_pcfg.pcfg_extraction()
    print('Extract pcfg Done')
if sys.argv[3] == 'True' :
    import OOV
    OOV.OOV_handling(sys.argv[4],
                     sys.argv[5],
                     sys.argv[6])
    print('Extract pcfg Done')

#####################################################################
#                           Load data                               #
#####################################################################
with codecs.open(os.path.join(sys.argv[8]+"\\", "PCFG_unary_dict.pkl"), 'rb') as file:
    unary_dict = pickle.load(file)
file.close()

with codecs.open(os.path.join(sys.argv[8]+"\\", "PCFG_binary_dict.pkl"), 'rb') as file:
    binary_dict = pickle.load(file)
file.close()

with codecs.open(os.path.join(sys.argv[8]+"\\", "PCFG_postags_dict.pkl"), 'rb') as file:
    postags_dict = pickle.load(file)
file.close()

with codecs.open(os.path.join(sys.argv[8]+"\\", "NT_set.pkl"), 'rb') as file:
    NT_set = pickle.load(file)
file.close()

with codecs.open(os.path.join(sys.argv[8]+"\\", "T_set.pkl"), 'rb') as file:
    T_set = pickle.load(file)
file.close()

with codecs.open(os.path.join(sys.argv[8]+"\\", "postags_set.pkl"), 'rb') as file:
    postags_set = pickle.load(file)
file.close()

with codecs.open(os.path.join(sys.argv[8]+"\\", "PCFG_unary_freq.pkl"), 'rb') as file:
    unary_freq = pickle.load(file)
file.close()

with codecs.open(os.path.join(sys.argv[8]+"\\", "PCFG_binary_freq.pkl"), 'rb') as file:
    binary_freq = pickle.load(file)
file.close()

with codecs.open(os.path.join(sys.argv[8]+"\\", "PCFG_postags_freq.pkl"), 'rb') as file:
    postags_freq = pickle.load(file)
file.close()


#####################################################################
#                               MAIN                                #
#####################################################################
parser = CYK_parser_class.CYK_parser()
parser.initialize(NT_set, T_set, postags_set,
                   unary_freq, binary_freq, postags_freq,
                   unary_dict, binary_dict, postags_dict)

parser.parse_corpus(input=sys.argv[8]+"\\"+'sequoia_test_corrected.txt', output=sys.argv[8]+"\\"+'output.txt')

#####################################################################
#                        Evaluation if needed                       #
#####################################################################
if sys.argv[7] == 'True':
    import evaluate
    evaluate.evaluation()
    print('Evaluation Done')