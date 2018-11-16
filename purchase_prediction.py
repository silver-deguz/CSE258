#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:09:48 2018

@author: jdeguzman
"""

import gzip
import pickle
import random
import numpy as np
import itertools
from collections import defaultdict
from collections import Counter
import time


def readGz(f):
    for l in gzip.open(f):
        yield eval(l)
        
# ------------------------------------------------------------------- #
# Build Training and Validation Set                                   #
# ------------------------------------------------------------------- #
def build_dataset(json_file):
    # ------------------------------------------------------------ #
    # Parse JSON to get user, item, and user_item pairs into lists #    
    # ------------------------------------------------------------ #
    user = []
    item = []
    user_item = set()

    for l in readGz(json_file):
        u, i = l['reviewerID'], l['itemID']
        user.append(u)
        item.append(i)
        user_item.add((u, i))
    
    # ---------------------- #
    # Negative Samples Pairs #
    # ---------------------- #
    neg_pairs = set()
    count = 0
    
    while (count < 100000):
        pair = (random.choice(user), random.choice(item))
        if pair not in user_item and pair not in neg_pairs:
            neg_pairs.add(pair)
            count += 1

    training = list(user_item)[:N]
    validation = list(user_item)[100000:] + list(neg_pairs)
    return training, validation


# ------------------------------------------------------------------- #
# Build Models                                                        #
# ------------------------------------------------------------------- #
def popularity_baseline(threshold = 0.4):
    item_count = defaultdict(int)
    for user, item in training_set:
        item_count[item] += 1   
    most_popular = [(item_count[x], x) for x in item_count]
    most_popular.sort(reverse=True)
    popular_items = set()
    total_purchases = len(training_set)
    count = 0
    for ic, i in most_popular:
        count += ic
        popular_items.add(i)
        if count > total_purchases*threshold: 
            break
    return popular_items
            
def purchase_history():
    Iu = defaultdict(list)
    Ui = defaultdict(list)
    count = 0
    for l in readGz("train.json.gz"):
        u, i = l['reviewerID'], l['itemID']
        Iu[u].append(i)
        Ui[i].append(u)
        # count only training data ratings and user-item pairs
        count = count + 1
        if count >= N:
            break
    return Iu, Ui


# ------------------------------------------------------------------- #
# Training Stage                                                      #
# ------------------------------------------------------------------- #
def itertools_chain(a):
    return set(itertools.chain.from_iterable(a))  

def jaccard_similarity(itemA, itemB):
    itemA_set = set(Ui[itemA])
    itemB_set = set(Ui[itemB])
    intersect = itemA_set.intersection(itemB_set)
    union = itemA_set.union(itemB_set)
    jacc_sim = len(intersect) / len(union)
    return jacc_sim

def training():
    for item in Ui:
        # for each user that bought this item, get all other items they've ever purchases
        itemlist = [ Iu[user] for user in Ui[item] ] 
        itemset = itertools_chain(itemlist) # convert items to a set
#        print(itemset)
        
        # compute similarity of each item in itemset to this item
        for item_k in itemset:
            S[(item, item_k)] = jaccard_similarity(item, item_k)
  

N = 200000 
training_set, validation_set = build_dataset('train.json.gz')
most_popular = popularity_baseline(threshold = 0.55)     
Iu, Ui = purchase_history() 
S = Counter()
training()
sim_average = sum(S.values()) / len(S)   # this is about 0.1398, make this threshold         

# ------------------------------------------------------------------- #
# Validation Stage                                                    #
# ------------------------------------------------------------------- #
def get_predictions(user, item):
    # No history of this user or item
    if user not in Iu:
        return False  
    
    sim_table = []
    similarity_threshold = 0.04
    for item_k in Iu[user]:
        if S[(item,item_k)] > similarity_threshold:
            sim_table.append(1)
        else: 
            sim_table.append(0)
#    sim_table = [1 if S[(item,item_k)] > similarity_threshold else 0 for item_k in Iu[user]]
    
    if sum(sim_table) >= 1:
        return True
    
    elif item in most_popular:
        return True
    
    else:
        return False
        
#    # build similarity table for this item compared to all items 
#    # all items user purchased in past
#    sim_table = []
#    for item_k in Iu[user]:
#        sim = jaccard_similarity(item, item_k)
#        sim_table.append(sim)
#        
#    if len(sim_table) == 0:
#        return False
#    print(max(sim_table))
#    if max(sim_table) >= 0.3:
#        return True
#    else:
#        return False

def get_accuracy(label, prediction):
    correct = [(a==b) for (a,b) in zip(label, prediction)]
    accy = sum(correct) / len(correct)
    return accy


#tic = time.time()
#groundtruth = ([1] * 100000) + ([0] * 100000)
#prediction_val = [1 if get_predictions(u,i) else 0 for u,i in validation_set]
#print(get_accuracy(groundtruth, prediction_val))
#toc = time.time()
#print(toc-tic)


# ------------------------------------------------------------------- #
# Kaggle Test Set Predictions                                         #
# ------------------------------------------------------------------- #
count = 0
predictions = open("predictions_Purchase.txt", 'w')
for l in open("pairs_Purchase.txt"):
    if l.startswith("reviewerID"):
        predictions.write(l)
        continue
    u, i = l.strip().split('-')

    if get_predictions(u,i):
        count += 1
        predictions.write(u + '-' + i + ",1\n")
    else:
        predictions.write(u + '-' + i + ",0\n")
predictions.close()
print(count)
