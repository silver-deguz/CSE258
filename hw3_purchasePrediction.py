#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:33:15 2018

@author: jdeguzman
"""
import gzip
import pickle
import random
import numpy as np
import itertools
from collections import defaultdict
#import sklearn.neighbors.distancemetric


def readGz(f):
    for l in gzip.open(f):
        yield eval(l)
        
# --------------------------------- #
# Build Training and Validation Set #
# --------------------------------- #
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
#    f = open('neg_samples.pckl', 'rb')
#    neg_pairs = pickle.load(f)
#    f.close() 
    neg_pairs = set()
    count = 0
    
    while (count < 100000):
        pair = (random.choice(user), random.choice(item))
        if pair not in user_item and pair not in neg_pairs:
            neg_pairs.add(pair)
            count += 1

    training = list(user_item)[:100000]
    validation = list(user_item)[100000:] + list(neg_pairs)
    return training, validation

# ------------------------- #
# Popularity Baseline Model #
# ------------------------- #
def popularity_baseline(threshold = 0.5):
    item_count = defaultdict(int)

    for user, item in training_set:
        item_count[item] += 1
#        item_count[item[1]] += 1
    
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


# ----------------------------------------------- #
# Question 1: Evaluate Accuracy of Validation Set #
# ----------------------------------------------- #
def get_accuracy(label, prediction):
    correct = [(a==b) for (a,b) in zip(label, prediction)]
    accy = sum(correct) / len(correct)
    return accy

#groundtruth = ([1] * 100000) + ([0] * 100000)
#popular_items = popularity_baseline()
#prediction = [1 if (i in popular_items) else 0 for u, i in validation_set]
#accuracy = get_accuracy(groundtruth, prediction)
#print(accuracy)


#predictions = open("predictions_Purchase.txt", 'w')
#for l in open("pairs_Purchase.txt"):
#    if l.startswith("reviewerID"):
#        predictions.write(l)
#        continue
#    u, i = l.strip().split('-')
#
#    if i in popular_items:
#        predictions.write(u + '-' + i + ",1\n")
#    else:
#        predictions.write(u + '-' + i + ",0\n")
#predictions.close()

# ------------------------------------ #
# Question 2: Finding Better Threshold #
# ------------------------------------ # 
#popular_items2 = popularity_baseline(threshold=0.6)
#prediction2 = [1 if (pair[1] in popular_items2) else 0 for pair in validation_set]
#best_accuracy = get_accuracy(groundtruth, prediction2)


# ----------------------------------- #
# Question 3: Category Baseline Model #
# ----------------------------------- # 
def getKey(item):
    return item[0]

def itertools_chain(a):
    return list(itertools.chain.from_iterable(a)) 

def build_trainingset(json_file):
    user_item_category = []
    for l in readGz(json_file):
        u, i, c = l['reviewerID'], l['itemID'], l['categories']
        user_item_category.append([u, i, c])
    return user_item_category[:200000]
            
def purchase_history_baseline(training_set):
    user_purchases = defaultdict(list)
    item_categories = defaultdict(list)
    for l in training_set:
        c = l[2]
        user_purchases[l[0]].append(c)
        item_categories[l[1]].append(c)
    return user_purchases, item_categories

def get_predictions(user, item):
    # No history of this user or item
    if (user not in user_purchases) or (item not in item_categories):
        return False  
    
    userset = itertools_chain(user_purchases[user])
    itemset = itertools_chain(item_categories[item])
#    intersect = userset.intersection(itemset)
    intersect = set(map(tuple, userset)).intersection(map(tuple, itemset))
    return len(intersect) != 0

#training_set, validation_set = build_dataset('train.json.gz')
#training_set2 = build_trainingset('train.json.gz')
#training_set2.sort(key=getKey)
#user_purchases, item_categories = purchase_history_baseline(training_set2)


# ------------------------------------------------- #
# Evaluating Accuracy of Q3 Model on Validation Set #
# ------------------------------------------------- #
#prediction_Q3 = [1 if get_predictions(pair[0], pair[1]) else 0 for pair in validation_set]
#print(get_accuracy(groundtruth, prediction_Q3))


# -------------------------------------- #
# Question 4: Kaggle Test Set Evaluation #
# -------------------------------------- # 
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




 
        
