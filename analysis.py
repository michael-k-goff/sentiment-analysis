# Imports
import pandas as pd
import numpy as np
import time
import decision_tree
import regression

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

def apply_model(master_list, master_dict, clf):
    # Apply the classifier to the test data
    test_results = [] # Predicted values
    test_values = [] # Actual values
    num_examples_to_test, batch_size = 0, 1000 # First value should be 0 for the full test set
    if num_examples_to_test == 0:
        num_examples_to_test = sum(1 for line in open('processed_acl/kitchen/unlabeled.review'))

    # Only do some of the lines for now
    with open("processed_acl/kitchen/unlabeled.review") as file:
        num_lines = 0
        df_batch = None
        for line in file:
            if (num_lines % batch_size) == 0 and num_lines < num_examples_to_test:
                df_batch = pd.DataFrame(0, np.arange(min(batch_size, num_examples_to_test-num_lines)), np.arange(len(master_list)))
            if (num_lines < num_examples_to_test):
                row_in_df = df_batch.iloc[num_lines%batch_size]
                words = line.rstrip().split(' ')
                for i in range(len(words)-1):
                    word = words[i].split(":")[0]
                    if word in master_dict:
                        row_in_df[master_dict[word]] = words[i].split(":")[1]
                test_values.append(words[-1].split("#")[2][1:])
            num_lines += 1
            if ((num_lines % batch_size) == 0 or num_lines == num_examples_to_test) and num_lines <= num_examples_to_test:
                predicted_values = clf.predict(df_batch)
                for i in range(len(df_batch.index)):
                    test_results.append(predicted_values[i])
    return test_values, test_results

def get_score(test_values, test_results):
    # Calculate accuracy
    den = len(test_values)
    num = len([0 for i in range(den) if test_values[i]==test_results[i]])
    print([num, den])
    print(num/den)
    
master_dict, master_list, number_examples = build_master_dict()
df_train, rating = regression.build_train_df(master_dict, master_list, number_examples)
clf = regression.train_model(df_train, rating)
test_values, test_results = apply_model(master_list, master_dict, clf)
get_score(test_values, test_results)
