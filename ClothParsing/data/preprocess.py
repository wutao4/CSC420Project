import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, '..')
import header


##########################################################################################
# Data Pre-processing: filter certain categories and split train/test sets
# Reference: Kaggle dataset tutorials:
#            https://www.kaggle.com/wangxin93/mastercategory-classification-senet
#            https://www.kaggle.com/marlesson/building-a-recommendation-system-using-cnn
##########################################################################################


DO_SPLIT = False  # Flag indicating whether to split the train/test dataset again
DATA_PATH = "../datasets/myntradataset/"
TRAINLIST_PATH = "../lists/train_list.txt"
TESTLIST_PATH = "../lists/test_list.txt"

# Read data in 'styles.csv' file. Neglect lines with error
df = pd.read_csv(os.path.join(DATA_PATH, 'styles.csv'), error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)
print("Data size before filtering:", df.shape)
# Plot data distribution
df.articleType.value_counts().sort_values().plot(kind='barh')
plt.show()

# Clean non-existing files and filter the clothing types we need
problems = []
for idx, row in df.iterrows():
    if row.articleType not in header.TYPES \
            or not os.path.exists(os.path.join(DATA_PATH, 'images', row['image'])):
        problems.append(idx)
df.drop(df.index[problems], inplace=True)
print("Data size after  filtering:", df.shape)
# Plot data distribution
df.articleType.value_counts().sort_values().plot(kind='barh')
plt.show()


# Split train and test dataset
if DO_SPLIT:
    train_df, test_df = train_test_split(df, test_size=0.1)
    print(train_df.shape, test_df.shape)
    f = open(TRAINLIST_PATH, 'w')
    for idx, row in train_df.iterrows():
        f.write(str(row.id) + '\n')
    f.close()
    f = open(TESTLIST_PATH, 'w')
    for idx, row in test_df.iterrows():
        f.write(str(row.id) + '\n')
    f.close()


if __name__ == '__main__':
    print("----------- start -------------")

    # Count the number of each type of articles in train/test set
    train, test = {}, {}
    for t in header.TYPES:
        train[t] = 0
        test[t] = 0
    df = df.set_index('id', drop=False)
    f = open(TRAINLIST_PATH, 'r')
    lines = f.readlines()
    for row_id in lines:
        row = df.loc[int(row_id.rstrip())]
        train[row['articleType']] += 1
    f.close()
    f = open(TESTLIST_PATH, 'r')
    lines = f.readlines()
    for row_id in lines:
        row = df.loc[int(row_id.rstrip())]
        test[row['articleType']] += 1
    f.close()
    print("Training set:")
    print(train)
    print("Test set:")
    print(test)

    print("------------ end --------------")

