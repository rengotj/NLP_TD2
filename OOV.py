# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:32:44 2019
@author: juliette rengot

A module that handles Out Of Vocabulary words
"""
#####################################################################
#                        IMPORT MODULES                             #
#####################################################################
import re
import codecs
import pickle
import numpy as np
from operator import itemgetter

import spelling_error_proba

#####################################################################
#                      Levenshtein distance                         #
#####################################################################
def Levenshtein(word1, word2,
                caracter_to_idx=None,reversal_use=False,
                unigram_proba=None, unigram_token_to_idx=None, proba_spell_model=None, prob_use=False,
                mle=None, previous_word=None, context_use=False):
    """ How many steps to transform word1 into word2 ?"""
    assert((not reversal_use) or 
           (reversal_use and type(caracter_to_idx)!=type(None)))
    assert((not prob_use) or
           (prob_use and type(unigram_proba)!=type(None) and type(unigram_token_to_idx)!=type(None) and type(proba_spell_model)!=type(None)))
    if type(previous_word)==type(None):
        context_use=False
    assert((not context_use) or
           (context_use and prob_use and type(mle)!=type(None)))
    
    rows = len(word1)+1
    cols = len(word2)+1

    if reversal_use :
        da = np.zeros(len(caracter_to_idx.keys()), dtype=np.int)
        
    dist = [[0 for x in range(cols)] for x in range(rows)]
    dist[1][0] = 1
    dist[0][1] = 1
    #deletion
    for i in range(2, rows):
        if prob_use and word1[i-2] in proba_spell_model.X and word1[i-1] in proba_spell_model.Y :
            idx1 = proba_spell_model.X.index(word1[i-2])
            idx2 = proba_spell_model.Y.index(word1[i-1])
            dist[i][0] = dist[i-1][0] + 1-proba_spell_model.Deletion_prob[idx1, idx2]
        else :
            dist[i][0] = i
    # insertion
    for i in range(2, cols):
        if prob_use and word2[i-2] in proba_spell_model.X and word2[i-1] in proba_spell_model.Y :
            idx1 = proba_spell_model.X.index(word2[i-2])
            idx2 = proba_spell_model.Y.index(word2[i-1])
            dist[0][i] = dist[0][i-1] + 1-proba_spell_model.Insertion_prob[idx1, idx2]
        else :
            dist[0][i] = i

    for col in range(1, cols):
        if reversal_use :
            db = 0
            
        for row in range(1, rows):
            if reversal_use :
                k = da[caracter_to_idx[word1[row-1]]]
                l = db
                
            if prob_use and word2[col-1] in proba_spell_model.X and word1[row-1] in proba_spell_model.Y :
                idx1 = proba_spell_model.X.index(word2[col-1])
                idx2 = proba_spell_model.Y.index(word1[row-1])               
                cost_deletion = 1-proba_spell_model.Deletion_prob[idx1, idx2]
            else :
                cost_deletion = 1
                
            if prob_use and word2[col-2] in proba_spell_model.X and word2[col-1] in proba_spell_model.Y :
                idx1 = proba_spell_model.X.index(word2[col-2])
                idx2 = proba_spell_model.Y.index(word2[col-1])    
                cost_insertion = 1-proba_spell_model.Insertion_prob[idx1, idx2]
            else :
                cost_insertion = 1
                
            if word1[row-1] == word2[col-1]:
                cost_substitution = 0
                if reversal_use :
                    db = col
            else:
                if prob_use and word1[row-1] in proba_spell_model.Y and word2[col-1] in proba_spell_model.Y :
                    idx1 = proba_spell_model.Y.index(word1[row-1])
                    idx2 = proba_spell_model.Y.index(word2[col-1])
                    cost_substitution = 1-proba_spell_model.Substitution_prob[idx1, idx2]
                else:
                    cost_substitution = 1
                    
            if reversal_use :    
                if prob_use and word1[row-1] in proba_spell_model.Y and word1[row-2] in proba_spell_model.Y :
                    idx1 = proba_spell_model.X.index(word1[row-1])
                    idx2 = proba_spell_model.Y.index(word1[row-2])    
                    cost_reversal = 1-proba_spell_model.Reversal_prob[idx1, idx2]
                else :
                    cost_reversal = 1

                dist[row][col] = min(dist[row-1][col]   + cost_deletion,     # deletion
                                     dist[row][col-1]   + cost_insertion,    # insertion
                                     dist[row-1][col-1] + cost_substitution, # substitution
                                     dist[max(0, k-1)][max(0, l-1)] + row-k-1 + cost_reversal + col-l-1) 
            else :
                dist[row][col] = min(dist[row-1][col]   + cost_deletion,     # deletion
                                     dist[row][col-1]   + cost_insertion,    # insertion
                                     dist[row-1][col-1] + cost_substitution) # substitution
        if reversal_use :
            da[caracter_to_idx[word1[row-1]]] = row

    if prob_use :
        if context_use and word2 in mle.keys() and previous_word in mle[word2].keys():
            return dist[row][col]*mle[word2][previous_word]*(10**7)
        
        assert(word2 in unigram_token_to_idx.keys())
        idx = unigram_token_to_idx[word2]
        return dist[row][col]*unigram_proba[idx]*(10**7)
    
    return dist[row][col]

########################################################################
#                      polyglot embedding                              #
#code adapted from https://nbviewer.jupyter.org/gist/aboSamoor/6046170 #
########################################################################
def case_normalizer(word, dictionary):
  """ In case the word is not available in the vocabulary,
     we can try multiple case normalizing procedure.
     We consider the best substitute to be the one with the lowest index,
     which is equivalent to the most frequent alternative."""
  w = word
  lower = (dictionary.get(w.lower(), 1e12), w.lower())
  upper = (dictionary.get(w.upper(), 1e12), w.upper())
  title = (dictionary.get(w.title(), 1e12), w.title())
  results = [lower, upper, title]
  results.sort()
  index, w = results[0]
  if index != 1e12:
    return w
  return word

def normalize(word, word_id, DIGITS):
    """ Find the closest alternative in case the word is OOV."""
    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = case_normalizer(word, word_id)

    if not word in word_id:
        return None
    return word

def l2_nearest(embeddings, word_index, k):
    """Sorts words according to their Euclidean distance.
       To use cosine distance, embeddings has to be normalized so that their l2 norm is 1."""

    e = embeddings[word_index]
    distances = (((embeddings - e) ** 2).sum(axis=1) ** 0.5)
    sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
    return zip(*sorted_distances[:k])


#####################################################################
#                           Main function                           #
#####################################################################
def OOV_handling(reversal_use_bool, prob_use_bool, context_use_bool) :

    #####################################################################
    #                           Load data                               #
    #####################################################################
    #Testing sentences
    with open("sequoia_test.txt", 'r', encoding='UTF-8') as file:
        corpus = file.read()
        corpus = corpus.splitlines()
    file.close()
    
    #Training sentences
    with open("sequoia_train.txt", 'r', encoding='UTF-8') as file:
        corpusTrain = file.read()
        unique_caracters = ''.join(set(corpusTrain))
        if reversal_use_bool :
            caracter_to_idx = {a:b for b,a in enumerate(unique_caracters)}
        else :
            caracter_to_idx = None
        corpusTrain = corpusTrain.splitlines()
    file.close()
    
    #training dictionnary
    with codecs.open("words_set.pkl", 'rb') as file:
        words_set = pickle.load(file)
    file.close()

    #####################################################################
    #                Create unigram model from training data            #
    #                                and                                #
    #       Create maximum likelihood estimate from training data       #
    #####################################################################
    if prob_use_bool :
        unigram_table = {'UNKNOWN':0}
        unigram_token_to_idx = {'UNKNOWN':0}
        unigram_idx_to_token = {0:'UNKNOWN'}
        
        if context_use_bool :
            mle_rate = 0.6
            mle = {} #maximum likelihood estimate
        else :
            mle=None
    
        idx = 1
        for i in range(len(corpusTrain)):
          sequence = corpusTrain[i].split()
          for j in range(len(sequence)):
              token = sequence[j]
              #unigram model
              if token in unigram_table.keys():
                  assert(token in unigram_token_to_idx.keys())
                  unigram_table[token] += 1
              else:
                  unigram_table[token] = 1
                  unigram_token_to_idx[token] = idx
                  unigram_idx_to_token[idx] = token
                  idx += 1
                  
              # maximum likelihood estimate
              if context_use_bool and j>0 :
                  if token in mle.keys():
                      if sequence[j-1] in mle[token].keys():
                          mle[token][sequence[j-1]] += 1
                      else:
                          mle[token][sequence[j-1]] = 1
                  else:
                    mle[token] = {}
                    mle[token][sequence[j-1]] = 1
        
        # convert into probabilities
        unigram_proba = np.array([unigram_table[token] for token in unigram_table.keys()])
        unigram_proba = unigram_proba/np.sum(unigram_proba)
        
        if context_use_bool :
            for i in mle.keys():
                for j in mle[i].keys() :
                    mle[i][j] = mle_rate*unigram_proba[unigram_token_to_idx[i]] + (1-mle_rate)*mle[i][j]/unigram_table[j]
    else :
        unigram_proba=None,
        unigram_token_to_idx=None
        mle=None

    #####################################################################
    #                      Create spelling error model                  #
    #####################################################################
    if prob_use_bool :
        proba_spell_model = spelling_error_proba.spelling_stats()
    else :
        proba_spell_model = None
    ########################################################################
    #                      polyglot embedding                              #
    #code adapted from https://nbviewer.jupyter.org/gist/aboSamoor/6046170 #
    ########################################################################
    with open("polyglot-fr.pkl", 'rb') as file:
        polyglot_words, polyglot_embeddings = pickle.load(file, encoding='latin1')
    file.close()
    
    polyglot_words = list(polyglot_words)
    #remove words in polyglot_words that are not in the training vocabulary
    for i in range(len(polyglot_words)-1, -1, -1):
        word = polyglot_words[i]
        if word not in words_set :
            polyglot_words.remove(word)
            polyglot_embeddings = np.delete(polyglot_embeddings, (i), axis=0)
    
    # Noramlize digits by replacing them with #
    DIGITS = re.compile("[0-9]", re.UNICODE)
    # Map words to indices and vice versa
    polyglot_word_id = {w:i for (i, w) in enumerate(polyglot_words)}
    polyglot_id_word = dict(enumerate(polyglot_words))


    #####################################################################
    #                        Word replacement                           #
    #####################################################################
    new_corpus = []
    for sentence in corpus :
        new_sentence = []
        word_list = sentence.split()
        for i in range(len(word_list)) :
            word = word_list[i]
            if i>1 and context_use_bool:
                previous_word = word_list[i-1]
            else :
                previous_word = None
                
            if word not in words_set :
                #Change of case
                word = word.lower()
                if word in words_set :
                    new_sentence.append(word)
    
                else :
                    success = False
                    for j in range(1,len(word)):
                        if word[:j] in words_set and word[j:] in words_set :
                            success = True
                            new_sentence.append(word[:j])
                            new_sentence.append(word[j:])
                            continue
                        elif word[:j]+'-'+word[j:] in words_set:
                            success = True
                            new_sentence.append(word[:j])
                            new_sentence.append(word[j:])
                            continue
                        
                    if not success :    
                        if '-' in word:
                            list_word = word.split('-')
                            while '' in list_word:
                                list_word.remove('')
                        else:
                            list_word = [word]
                            
                        for word in list_word :    
                            if word in words_set :
                                new_sentence.append(word)
                            else:                
                                candidate = []
                                candidate_dist = []
                                print('unknown word : ', word)
                                # look for genuine unknown words
                                word_norm = normalize(word, polyglot_word_id, DIGITS)
                                if not word_norm:
                                    print("Out of polyglot vocabulary")
                                else :
                                    indices, distances = l2_nearest(polyglot_embeddings,
                                                                    polyglot_word_id[word_norm],
                                                                    2)
                                    candidate.append(polyglot_id_word[indices[1]])
                                    candidate_dist.append([distances[1]])
                                    
                                # look for spelling error
                                for train_word in words_set:
                                    cost = Levenshtein(word, train_word, 
                                                       caracter_to_idx, reversal_use_bool,
                                                       unigram_proba, unigram_token_to_idx, proba_spell_model, False,
                                                       mle, previous_word, False)
        
                                    if prob_use_bool and cost<=2:
                                        candidate.append(train_word)
                                        dist = Levenshtein(word, train_word, 
                                                           caracter_to_idx, reversal_use_bool,
                                                           unigram_proba, unigram_token_to_idx, proba_spell_model, prob_use_bool,
                                                           mle, previous_word, context_use_bool)
                                        candidate_dist.append(dist)
                                    elif cost<=2 :
                                        candidate.append(train_word)
                                        candidate_dist.append(cost)
                                        
                                if len(candidate_dist)>0 :
                                    idx = np.argmin(candidate_dist)
                                    chosen_word = candidate[idx]
                                    new_sentence.append(chosen_word)
                                    print('candidates : ', candidate)
                                    print('chosen_word : ', chosen_word)
                                else :
                                    print('Choose special token UNKNOWN')
                                    new_sentence.append("<UNKNOWN>")
            else :
                new_sentence.append(word)
        new_corpus.append(new_sentence)
    
    fileOut = codecs.open("sequoia_test_corrected.txt", 'w', 'UTF-8')
    for new_sentence in new_corpus:
        fileOut.write(" ".join(new_sentence))
        fileOut.write('\n')
    fileOut.close()
    
    return()

if __name__ == '__main__' :
    OOV_handling(True, True, True)