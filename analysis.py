# To be done:
# Write the function below to handle arbitrary batch sizes and number of lines to be processed

import pandas as pd
import numpy as np
from sklearn import tree
import time

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
    return master_dict, number_examples

master_dict, number_examples = build_master_dict()

# A list of the words and in the dictionary, keep track of the index for constant time access
master_list = list(master_dict.keys())
for i in range(len(master_list)):
    master_dict[master_list[i]] = i
    
# The training dataframe
num_training_examples = number_examples["negative"]+number_examples["positive"]
df_train = pd.DataFrame(0, index=np.arange(num_training_examples), columns = np.arange(len(master_list)))
rating = []

def add_words(filename, start_point):
    with open(filename) as file:
        row_count = start_point
        for line in file:
            row_in_df = df_train.iloc[row_count]
            words = line.rstrip().split(' ')
            for i in range(len(words)-1):
                word = words[i].split(":")[0]
                count = int(words[i].split(":")[1])
                row_in_df[master_dict[word]] = count
            rating.append(words[-1].split("#")[2][1:])
            row_count += 1
            
add_words("processed_acl/kitchen/negative.review",0)
add_words("processed_acl/kitchen/positive.review",number_examples["negative"])

# Train the classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(df_train, rating)

# Apply the classifier to the test data
test_results = [] # Predicted values
test_values = [] # Actual values
num_examples_to_test, batch_size = 0, 80
if num_examples_to_test == 0:
    num_examples_to_test = sum(1 for line in open('processed_acl/kitchen/unlabeled.review'))

# Only do some of the lines for now
start_time = time.time()
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
end_time = time.time()
print("Time: "+str(end_time-start_time))
            
# Calculate accuracy
den = len(test_values)
num = len([0 for i in range(den) if test_values[i]==test_results[i]])
print([num, den])
print(num/den)