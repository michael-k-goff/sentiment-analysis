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
num_examples_to_test, batch_size = 50, 50

# Only do some of the lines for now
num_lines = 0
start_time = time.time()
with open("processed_acl/kitchen/unlabeled.review") as file:
    df_batch = pd.DataFrame(0, np.arange(batch_size), np.arange(len(master_list)))
    for line in file:
        if (num_lines < num_examples_to_test):
            row_in_df = df_batch.iloc[num_lines]
            if (num_lines % 10 == 0): # Tracker
                print(num_lines)
            # word_vector = np.zeros(len(master_list)) # Essentially a batch size of 1
            words = line.rstrip().split(' ')
            for i in range(len(words)-1):
                word = words[i].split(":")[0]
                if word in master_dict:
                    #word_vector[master_dict[word]] = words[i].split(":")[1]
                    row_in_df[master_dict[word]] = words[i].split(":")[1]
            test_values.append(words[-1].split("#")[2][1:])
            #predicted_value = clf.predict(pd.DataFrame(word_vector).T)[0]
            #test_results.append(predicted_value)
        num_lines += 1
    predicted_values = clf.predict(df_batch)
    for i in range(batch_size):
        test_results.append(predicted_values[i])
end_time = time.time()
print("Time: "+str(end_time-start_time))
            
# Calculate for
den = len(test_values)
num = len([0 for i in range(den) if test_values[i]==test_results[i]])
print([num, den])
print(num/den)