import numpy as np

############################################
#      Header File: global variables       #
############################################
HELLO = "Hello World!"


# In this project we consider the following clothing types from the entire dataset

# Article types
TYPES = ["Caps",
         "Shirts", "Tops", "Sweatshirts", "Jackets",
         "Trousers", "Jeans", "Shorts", "Track Pants", "Skirts",
         "Casual Shoes", "Sports Shoes", "Flip Flops", "Sandals",
         "Backpacks"]

# Article types and corresponding indices
idx = 0
TYPE_INDEX = {}
for t in TYPES:
    TYPE_INDEX[t] = idx
    idx += 1

# Article types and corresponding one-hot vector
TYPE_LABEL = {}
zero = np.zeros(len(TYPES))
for key in TYPE_INDEX:
    onehot = zero.copy()
    onehot[TYPE_INDEX[key]] = 1
    TYPE_LABEL[key] = onehot
