# A bag of words model applied to a dataset of reviews of kitchen products.
# Data can be downloaded here: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
# For more information, see https://www.projectpro.io/article/sentiment-analysis-project-ideas-with-source-code/518

# Two basic learning models are implemented, decision trees and linear regression.
# See the respective modules for those.

# Imports
import pandas as pd
import numpy as np
import time
import decision_tree
import regression

# Of the two models, the regression seems to perform better, with around 86% accuracy on the test set.
# The decision tree gets accuracy of around 74%.

module = decision_tree # Set to either 'decision_tree' or 'regression' based on the desired model.

# A dictionary containing the word vectors for all training examples
def build_master_dict():
    master_dict = {}

    def add_to_master_list(filename):
        num_lines = 0
        with open(filename) as file:
            for line in file:
                num_lines += 1
                words = line.rstrip().split(' ')
                for i in range(len(words)-1):
                    word = words[i].split(":")[0]
                    if word not in master_dict:
                        master_dict[word] = 1
        return num_lines

    number_examples = {}
    number_examples["negative"] = add_to_master_list("processed_acl/kitchen/negative.review")
    number_examples["positive"] = add_to_master_list("processed_acl/kitchen/positive.review")
    
    # Put all the items in a list as well for fast access
    master_list = list(master_dict.keys())
    for i in range(len(master_list)):
        master_dict[master_list[i]] = i
    
    return master_dict, master_list, number_examples

def get_score(test_values, test_results):
    # Calculate accuracy
    den = len(test_values)
    num = len([0 for i in range(den) if test_values[i]==test_results[i]])
    print("Got "+str(num)+" right out of "+str(den)+" test.")
    print("Accuracy: "+ str(num/den))
    
master_dict, master_list, number_examples = build_master_dict()
df_train, rating = module.build_train_df(master_dict, master_list, number_examples)
clf = module.train_model(df_train, rating)
test_values, test_results = module.apply_model(master_list, master_dict, clf)
get_score(test_values, test_results)
