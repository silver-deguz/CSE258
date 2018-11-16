#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:29:00 2018

@author: jdeguzman
"""
import gzip
import random
import numpy as np
import itertools
from collections import defaultdict
from collections import Counter
from sklearn.metrics import mean_squared_error


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
    user_item = []
    groundtruth = []

    for l in readGz(json_file):
        u, i, r = l['reviewerID'], l['itemID'], l['rating']
        user_item.append((u, i))
        groundtruth.append(r)

    training = user_item[:100000]
    validation = user_item[100000:]
    groundtruth_training = groundtruth[:100000]
    groundtruth_validation = groundtruth[100000:]
    return training, validation, groundtruth_training, groundtruth_validation


# ----------------------------------------- #
# Question 5: Trivial Predictor Performance #
# ----------------------------------------- #
#training_set, validation_set, groundtruth_train, groundtruth_val = build_dataset('train.json.gz')
#R = Counter()
#Iu = defaultdict(list)
#Ui = defaultdict(list)
#
#count = 0
#for l in readGz("train.json.gz"):
#    u, i, r = l['reviewerID'], l['itemID'], l['rating']
#    R[(u,i)] = r
##    all_ratings.append(r)
#    
#    Iu[u].append(i)
#    Ui[i].append(u)
#    # count only training data ratings and user-item pairs
#    count = count + 1
#    if count >= 100000:
#        break



## This verifies the count of total purchased items and total users that 
## purchased items should be equal to 100000 each because there are 100000 
## pairs in training data
#total_items = [len(val) for key,val in Iu.items()]
#total_users = [len(val) for key,val in Ui.items()]
#print(sum(total_items))
#print(sum(total_users))


#global_average = sum(R.values()) / len(R)
#prediction = [global_average] * 100000
#mse_trivial = mean_squared_error(groundtruth_val, prediction) 
    

# ------------------------------------------------------- #
# Question 6: Fit Predictor of mean, user_bias, item_bias #
# ------------------------------------------------------- #
def update_terms(userBias, itemBias, lamb):
    # update alpha 
    N = len(training_set)
    avg_k = [ (R[u,i] - (userBias[u] + itemBias[i])) for u,i in training_set ]
    avg = sum(avg_k)/N
    
    # update beta_u
    for u in Iu:
        userBias_k = [ (R[u,i] - (avg + itemBias[i])) for i in Iu[u] ]
        userBias[u] = sum(userBias_k) / ( lamb + len(Iu[u]) ) 
   
    # update beta_i
    for i in Ui:
        itemBias_k = [ (R[u,i] - (avg + userBias[u])) for u in Ui[i] ]
        itemBias[i] = sum(itemBias_k) / ( lamb + len(Ui[i]) ) 
    return avg, userBias, itemBias


def objfunc(avg, userBias, itemBias, lamb):
    error = [ ( avg + userBias[u] + itemBias[i] - R[u,i] )**2 for u,i in training_set ]
    userBias_reg = [ userBias[u]**2 for u in userBias ]
    itemBias_reg = [ itemBias[i]**2 for i in itemBias ] 
    f = sum(error) + lamb * (sum(userBias_reg) + sum(itemBias_reg))
    print(f)
    return f


# --------------------------------------------- #
# Optimization using Iterative Gradient Descent #
# --------------------------------------------- #
#avg = 0
#userBias = defaultdict(int)
#itemBias = defaultdict(int)
#
#for u in Iu:
#    userBias[u] = 0    
#for i in Ui:
#    itemBias[i] = 0
#
#iteration = 0
#lamb = 1
#while iteration < 10:
#    avg, userBias, itemBias = update_terms(userBias, itemBias, lamb)
#    f = objfunc(avg, userBias, itemBias, lamb)
#    iteration += 1
#    predict_val = [ (avg + userBias[u] + itemBias[i]) for u,i in validation_set ]
#    new_mse = mean_squared_error(groundtruth_val, predict_val) 
#    print(new_mse)
#
#avg_final = avg
#userBias_final = userBias
#itemBias_final = itemBias

# ----------------------------- #
# Rating Prediction on Test Set #
# ----------------------------- #
#predictions = open("predictions_Rating.txt", 'w')
#for l in open("pairs_Rating.txt"):
#    if l.startswith("reviewerID"):
#        #header
#        predictions.write(l)
#        continue
#    u,i = l.strip().split('-')
#    
#    predict = np.clip((avg_final + userBias_final[u] + itemBias_final[i]), a_min = 1, a_max = 5)
#    predictions.write(u + '-' + i + ',' + str(predict) + '\n')
#predictions.close()


