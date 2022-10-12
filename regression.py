# As of October 11, 2022, this is not predicting properly. Fix that later.

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def train_model(df_train, rating):
    clf = LinearRegression().fit(df_train, rating)
    return clf
    
# The training dataframe
def build_train_df(master_dict, master_list, number_examples):
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
    
    for i in range(len(rating)):
        if rating[i] == 'negative':
            rating[i] = 0
        else:
            rating[i] = 1
    
    return df_train, rating