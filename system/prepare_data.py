# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:58:51 2019
@author: juliette rengot
A module to load an prepare initial corpus sequoia-corpus+fct.mrg_strict
"""
#####################################################################
#                        IMPORT MODULES                             #
#####################################################################
import os
import codecs
import re
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

def data_preparation():
    #####################################################################
    #                        Load the Corpus                            #
    #####################################################################
    data_file = codecs.open("sequoia-corpus+fct.mrg_strict", 'r', 'UTF-8')
    data = data_file.read().splitlines()
    N = len(data)
    
    #Ignore functional labels
    for i in range(N):
        compiled = re.compile(r"(?<=\()[A-Za-z_+^\-]+\-[^ ]+")             # Compile a regular expression pattern
        data[i] = compiled.sub(lambda x: x.group().split('-')[0], data[i]) # reject - and what follows
    data_file.close()
    
    #####################################################################
    #                          Training data                            #
    #####################################################################
    # Select Training data and save it in sequoia_train.tb file
    corpusTrain = data[:int(N*0.8)] #80% of data
    fileTrain = codecs.open("sequoia_train.tb", 'w', 'UTF-8')
    for s in corpusTrain:
        fileTrain.write(u"{0}\n".format(s))
    fileTrain.close()
    
    # Read parenthesis-delineated parse trees
    # and save the natural sentences in sequoia_train.txt
    fileOut = codecs.open("sequoia_train.txt", 'w', 'UTF-8')
    for tree in BracketParseCorpusReader("", "sequoia_train.tb").parsed_sents():
        fileOut.write(u"{0}\n".format(u" ".join(tree.leaves())))
    fileOut.close()
    
    #####################################################################
    #                        Development data                           #
    #####################################################################
    # Select Developmen data and save it in sequoia_dev.tb file
    corpusDev = data[int(N*0.8):int(N *0.9)] #10% of data
    fileDev = codecs.open("sequoia_dev.tb",'w', 'UTF-8')
    for s in corpusDev:
        fileDev.write(u"{0}\n".format(s))
    fileDev.close()
    
    # Read parenthesis-delineated parse trees
    # and save the natural sentences in sequoia_train.txt
    fileOut = codecs.open(os.path.join("sequoia_dev.txt"), 'w', 'UTF-8')
    for tree in BracketParseCorpusReader("", "sequoia_dev.tb").parsed_sents():
        fileOut.write(u"{0}\n".format(u" ".join(tree.leaves())))
    fileOut.close()
    
    #####################################################################
    #                           Testing data                            #
    #####################################################################
    # Select Testing data and save it in sequoia_test.tb file
    corpusTest = data[int(N*0.9):]
    fileTest = codecs.open("sequoia_test.tb", 'w', 'UTF-8')
    for s in corpusTest:
        fileTest.write(u"{0}\n".format(s))
    fileTest.close()
    
    # Read parenthesis-delineated parse trees
    # and save the natural sentences in sequoia_train.txt
    fileOut = codecs.open("sequoia_test.txt", 'w', 'UTF-8')
    for tree in BracketParseCorpusReader("", "sequoia_test.tb").parsed_sents():
        fileOut.write(u"{0}\n".format(u" ".join(tree.leaves())))
    fileOut.close()
    
    fileTest2 = codecs.open("sequoia_test_tree.txt", 'w', 'UTF-8')
    for s in corpusTest:
        fileTest2.write(u"{0}\n".format(s[2:-1]))
    fileTest2.close()
    
    return()

if __name__ == '__main__' :
    data_preparation()